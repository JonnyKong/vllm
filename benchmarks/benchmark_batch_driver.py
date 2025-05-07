# SPDX-License-Identifier: Apache-2.0
import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch
from benchmark_utils import get_gpu_name, get_result_root
from paths import RESULT_ROOT

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import (get_preselected_freq,
                                       nvml_get_available_freq,
                                       uniform_sample_sorted)
from vllm.utils import FlexibleArgumentParser


def gen_benchmark_idle_power_args():
    """
    Introduce a delay before issuing each batch to assess whether the power
    consumption between batches aligns with the idle power measured offline.
    """
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), 16)
    prefill_input_len = 1024
    prefill_bs = 2
    decode_input_len = 1024
    decode_bs = 256

    args = []
    for delay_time_s in [2.0]:
        for freq in test_freqs:
            expr_dir = (get_result_root() / 'request_timing' /
                        '2025-02-02_benchmark-idle-power' /
                        f'{get_gpu_name()}-pp1-tp1-delay{delay_time_s}')
            log_dir = expr_dir / \
                f'prefill-len-{prefill_input_len}-bs-{prefill_bs}_decode-len-{decode_input_len}-bs-{decode_bs}_freq-{freq}'
            args.append(
                BenchmarkBatchParam(
                    prefill_input_lens=[prefill_input_len] * prefill_bs,
                    decode_input_lens=[decode_input_len] * decode_bs,
                    log_dir=str(log_dir),
                    gpu_freq_mhz=freq,
                    delay_time_min_s=delay_time_s,
                    delay_time_max_s=delay_time_s,
                ))
    return args


def gen_sarathi_args():
    """
    Derive the SLO with Sarathi-serve's definition:

    "We define the SLO on P99 TBT to be equal to 5× and 25× the execution time
    of a decode iteration for a request ,with prefill length of 4k and 32 batch
    size) running without any prefill interference"
    """
    decode_bs = 32
    decode_input_len = 4000
    freq = max(nvml_get_available_freq())
    log_dir = (get_result_root() / 'request_timing' /
               '2025-02-26_benchmark-slo' / f'{get_gpu_name()}-pp1-tp1' /
               f'decode-len-{decode_input_len}-bs-{decode_bs}_freq-{freq}')
    # Use uneven batch sizes, which matches online serving
    decode_input_lens = list(range(decode_input_len, decode_input_len + 32))
    return BenchmarkBatchParam(
        prefill_input_lens=[],
        decode_input_lens=decode_input_lens,
        log_dir=str(log_dir),
        gpu_freq_mhz=freq,
        delay_time_min_s=0.0,
        delay_time_max_s=0.0,
    )


def gen_from_trace(
    gpu: str,
    model: str,
    skip_existing: bool = True,
):
    log_dir_base = (
        RESULT_ROOT /
        f'request_timing/2025-05-05_lat-model-profiling/{gpu}_{model}')

    # Batch shape traces are unchanged across GPU and models
    BATCH_SHAPE_TRACES: list[Path] = [
        RESULT_ROOT /
        'request_timing/2025-05-05_batch-shape-profiling/A40_Llama-3.1-8B-Instruct_qps9_reqs20000_fixed1740/perf_metric_3321721.csv',
        RESULT_ROOT /
        'request_timing/2025-05-05_batch-shape-profiling/A40_Llama-3.1-8B-Instruct_qps5_reqs20000_fixed1740/perf_metric_3378238.csv',
    ]

    max_counts: dict = {
        'prefill-only': 1000,
        'decode-only': 10000,
        'hybrid': 20000,
    }

    counter = Counter()
    params = []
    test_freqs = get_preselected_freq(gpu)

    for trace in BATCH_SHAPE_TRACES:
        df = pd.read_csv(trace)

        for idx, row in df.iterrows():
            num_computed_tokens = eval(
                row['num_precomputed_tokens_per_req_iter'])
            chunk_sizes = eval(row['chunk_size_per_req_iter'])

            # Skip over heartbeat rows
            if (len(num_computed_tokens) == 0):
                continue

            prefill_lens = []
            prefill_computed_lens = []
            decode_lens = []

            start_decode_ind = 0

            for i, size in enumerate(chunk_sizes):
                if size > 1:
                    start_decode_ind = i + 1

            for i in range(start_decode_ind):
                prefill_lens.append(num_computed_tokens[i] + chunk_sizes[i])
                prefill_computed_lens.append(num_computed_tokens[i])

            for i in range(start_decode_ind, len(chunk_sizes)):
                decode_lens.append(num_computed_tokens[i] + chunk_sizes[i])

            freq = test_freqs[idx % len(test_freqs)]
            p = BenchmarkBatchParam(
                prefill_input_lens=prefill_lens,
                prefill_completed_input_lens=prefill_computed_lens,
                decode_input_lens=decode_lens,
                log_dir=
                f"{log_dir_base}/logs/batch_{len(params):06d}_freq{freq}",
                gpu_freq_mhz=freq,
                min_num_iters=2,
                min_seconds=1,
            )

            # Apply upper limit to each type
            batch_type = p.get_batch_type()
            if counter[batch_type] < max_counts[batch_type]:
                params.append(p)
                counter[batch_type] += 1

    # Supplement prefills by extracting from hybrid batches if needed
    prefills_supp = []
    for p in params:
        if p.get_batch_type() != 'hybrid':
            continue
        if counter['prefill-only'] >= max_counts['prefill-only']:
            break
        p_copy = copy.deepcopy(p)
        p_copy.decode_input_lens.clear()
        p_copy.log_dir = f"{log_dir_base}/logs/batch_{(len(params) + len(prefills_supp)):06d}_freq{p.gpu_freq_mhz}"  # noqa
        prefills_supp.append(p_copy)
        counter['prefill-only'] += 1
    params.extend(prefills_supp)

    print('Batches per type: ', counter)

    if skip_existing:
        params = [p for p in params if not Path(p.log_dir).exists()]
    return params


def main(expr_fn: Callable, model: str):
    vllm_args = (f"--model {model} "
                 "-tp 1 -pp 1 "
                 "--disable-async-output-proc "
                 "--disable-python-gc "
                 "--enable-chunked-prefill "
                 "--collect-detailed-traces worker,power ")

    # Keep it same with `benchmark_serving_driver.sh`
    gpu_name = get_gpu_name()
    if gpu_name == 'A40' and model == 'meta-llama/Llama-3.1-8B-Instruct':
        vllm_args += ('--max-model-len 65536 --max-num-seqs 1024 '
                      '--max-num-batched-tokens 1024 ')
    elif gpu_name == 'T4' and model == 'microsoft/phi-2':
        vllm_args += '--max-model-len 2048 --dtype=half '
    elif gpu_name == 'A100-SXM4-80GB' and model == 'google/gemma-2-27b-it':
        vllm_args += ('--max-model-len 8192 --max-num-seqs 1024 '
                      '--max-num-batched-tokens 1024 ')
    else:
        raise NotImplementedError(f'gpu: {gpu_name}, model: {model}')
    print('vllm_args: ', vllm_args)

    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args.split())

    # Pass in a list instead of generator so tqdm prints progress
    params = expr_fn(get_gpu_name(), model)

    uvloop.run(benchmark_batch(vllm_args, params))


if __name__ == '__main__':
    expr_fn = {
        # 'idle-power': gen_benchmark_idle_power_args,
        # 'sarathi-serve-sla': gen_sarathi_args,
        'trace': gen_from_trace,
    }[sys.argv[1]]
    model = sys.argv[2]
    main(expr_fn, model)
