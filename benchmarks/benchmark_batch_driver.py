# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import itertools
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch
from benchmark_utils import (get_gpu_name, get_result_root,
                             uniform_sample_sorted)
from latency_and_power_model_sampler import (
    gen_benchmark_batch_args_sample_decode_only,
    gen_benchmark_batch_args_sample_hybrid,
    gen_benchmark_batch_args_sample_prefill_only)

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import nvml_get_available_freq
from vllm.utils import FlexibleArgumentParser


def gen_benchmark_batch_args(pp: int, tp: int, skip_existing: bool = False):
    expr_dir = (
        get_result_root() /
        f'request_timing/2025-02-02_benchmark-batch_llama70b/{get_gpu_name()}-pp{pp}-tp{tp}'
    )

    prefill_input_lens = [2048]
    prefill_bss = [1]
    decode_input_lens = [1024, 16384]
    decode_bss = [256]
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), 8)

    args = []
    for prefill_input_len, prefill_bs, decode_input_len, decode_bs, freq in \
            itertools.product(
                prefill_input_lens,
                prefill_bss,
                decode_input_lens,
                decode_bss,
                test_freqs,
            ):
        if prefill_bs == 0 and decode_bs == 0:
            continue

        log_dir = expr_dir / \
            f'prefill-len-{prefill_input_len}-bs-{prefill_bs}_decode-len-{decode_input_len}-bs-{decode_bs}_freq-{freq}'
        if skip_existing and os.path.exists(log_dir):
            continue

        args.append(
            BenchmarkBatchParam(
                prefill_input_lens=[prefill_input_len] * prefill_bs,
                decode_input_lens=[decode_input_len] * decode_bs,
                log_dir=str(log_dir),
                gpu_freq_mhz=freq,
            ))
    return args


def gen_benchmark_idle_power_args(pp: int, tp: int):
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
                        f'{get_gpu_name()}-pp{pp}-tp{tp}-delay{delay_time_s}')
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


def gen_sarathi_args(pp: int, tp: int):
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
               '2025-02-26_benchmark-slo' / f'{get_gpu_name()}-pp{pp}-tp{tp}' /
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


def gen_power_profiling_args(pp: int,
                             tp: int,
                             skip_existing: bool = True,
                             num_freqs: int = 11,
                             batch_type: Optional[str] = None):
    expr_dir = (
        get_result_root() /
        f'request_timing/2025-03-14_power-model-profiling/{get_gpu_name()}-pp{pp}-tp{tp}_llama8-3b'
    )

    # Since we are swapping out `log_dir`, pass in `skip_existing=False` for
    # sub-generators, and check for existence in this function
    args_dict = {
        'hybrid':
        gen_benchmark_batch_args_sample_hybrid(num_samples=10000,
                                               num_freqs=num_freqs),
        'prefill-only':
        gen_benchmark_batch_args_sample_prefill_only(
            num_samples=5000,
            num_freqs=num_freqs,
        ),
        'decode-only':
        gen_benchmark_batch_args_sample_decode_only(num_samples=5000,
                                                    num_freqs=num_freqs),
    }
    args_all = []
    for batch_type_, args in args_dict.items():
        if batch_type and batch_type_ != batch_type:
            continue
        for i, arg in enumerate(args):
            # Rewrite `args` as needed
            sample_name = f'{batch_type_}_{i:06d}_freq{arg.gpu_freq_mhz}_{hash(arg)}'  #noqa
            arg.log_dir = str(expr_dir / sample_name)
            args_all.append(arg)
    if skip_existing:
        args_all = [arg for arg in args_all if not Path(arg.log_dir).exists()]
    return args_all


def gen_args_test_energy_linearity_of_hybrid_batches(pp: int,
                                                     tp: int,
                                                     skip_existing: bool = True,
                                                     num_freqs: int = 11):
    """
    For each hybrid batch, break it down in to prefill-only and decode-only
    batches, to see if energy of the hybrid batch is the sum of prefill and
    decode batches.
    """
    expr_dir = (
        get_result_root() /
        f'request_timing/2025-03-30_test_energy_linearity_of_hybrid_batches/{get_gpu_name()}-pp{pp}-tp{tp}_llama8-3b'
    )
    args = gen_power_profiling_args(pp,
                                    tp,
                                    skip_existing=False,
                                    num_freqs=num_freqs,
                                    batch_type='hybrid')
    ret = []
    for arg in args:
        arg_prefill = copy.deepcopy(arg)
        arg_prefill.decode_input_lens = []
        arg_prefill.log_dir = str(
            expr_dir / Path(arg.log_dir).stem) + '_prefill-only-part'
        if not (skip_existing and Path(arg_prefill.log_dir).exists()):
            ret.append(arg_prefill)

        arg_decode = copy.deepcopy(arg)
        arg_decode.prefill_input_lens = []
        arg_decode.log_dir = str(
            expr_dir / Path(arg.log_dir).stem) + '_decode-only-part'
        if not (skip_existing and Path(arg_decode.log_dir).exists()):
            ret.append(arg_decode)
    return ret


def gen_chunked_prefill_args(tp: int,
                             pp: int,
                             skip_existing: bool = False,
                             num_freqs: int = 11):
    params = []
    decode_bss = [0, 100, 200, 300, 400]
    decode_len = 200
    for decode_bs in decode_bss:
        decode_lens = [decode_len] * decode_bs
        for i in range(5):
            prefill_lens = [1024 * (i + 1)]
            prefill_completed_lens = [1024 * i]
            params.append(
                BenchmarkBatchParam(
                    prefill_input_lens=prefill_lens,
                    prefill_completed_input_lens=prefill_completed_lens,
                    decode_input_lens=decode_lens,
                    log_dir=
                    "/home/hemingxuan0926/datla/vllm/benchmarks/tmp/" + \
                    f"results_{prefill_lens[0]}_{decode_bs}",
                    gpu_freq_mhz=1125,
                    min_num_iters=2,
                    min_seconds=1,
                ))
    return params


def gen_from_trace(
    tp: int,
    pp: int,
    start_sample: int = 0,
    end_sample: int = 20000,
    num_freqs: int = 11,
    trace_dir:
    str = "/export2/datla/energy_efficient_serving_results/" \
        "azure_trace_sampling/slightly_underloaded_qps/logs"
):

    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), num_freqs)
    params = []
    csv_files = [
        f for f in os.listdir(trace_dir)
        if re.match(r'perf_metric_\d+\.csv', f)
    ]

    for filename in csv_files:
        full_path = os.path.join(trace_dir, filename)
        df = pd.read_csv(full_path)

        count = -1
        for idx, row in df.iterrows():

            # Convert stringified list to actual list
            num_computed_tokens = ast.literal_eval(
                row['num_precomputed_tokens_per_req_iter'])
            chunk_sizes = ast.literal_eval(row['chunk_size_per_req_iter'])

            if (len(num_computed_tokens) == 0):
                continue

            count += 1

            if (count < start_sample):
                continue

            if (count >= end_sample):
                break

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

            params.append(
                BenchmarkBatchParam(
                    prefill_input_lens=prefill_lens,
                    prefill_completed_input_lens=prefill_computed_lens,
                    decode_input_lens=decode_lens,
                    log_dir=
                    "/export2/datla/energy_efficient_serving_results/" \
                    "azure_trace_sampling/slightly_underloaded_qps_batches/" \
                    f"logs/batch_{count}",
                    gpu_freq_mhz=random.choice(test_freqs),
                    min_num_iters=2,
                    min_seconds=1,
                ))
    return params


def main(expr_fn: Callable):
    tp = 1
    pp = 1
    model = 'meta-llama/Llama-3.1-8B-Instruct'
    vllm_args = (f"--model {model} "
                 f"-tp {tp} "
                 f"-pp {pp} "
                 "--disable-async-output-proc "
                 "--max-num-seqs 1024 --max-num-batched-tokens 1024 "
                 "--max-model-len 65536 "
                 "--collect-detailed-traces worker").split()
    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args)

    # Pass in a list instead of generator so tqdm prints progress
    params = expr_fn(tp=tp, pp=pp)
    latencies = []

    uvloop.run(benchmark_batch(vllm_args, params, latencies))


if __name__ == '__main__':
    expr_fn = {
        'batch': gen_benchmark_idle_power_args,
        'idle-power': gen_benchmark_idle_power_args,
        'sarathi-serve-sla': gen_sarathi_args,
        'power_profiling': gen_power_profiling_args,
        'power_profiling_test_linearity_of_hybrid_batches':
        gen_args_test_energy_linearity_of_hybrid_batches,
        'chunked_prefill': gen_chunked_prefill_args,
        'trace': gen_from_trace,
    }[sys.argv[1]]
    main(expr_fn)
