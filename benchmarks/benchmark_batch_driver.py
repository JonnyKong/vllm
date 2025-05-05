# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import itertools
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch
from benchmark_utils import get_gpu_name, get_result_root
from latency_and_power_model_sampler import (
    gen_benchmark_batch_args_sample_decode_only,
    gen_benchmark_batch_args_sample_hybrid,
    gen_benchmark_batch_args_sample_prefill_only)

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import (nvml_get_available_freq,
                                       uniform_sample_sorted)
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
                             num_freqs: int = 10,
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
                                                     num_freqs: int = 10):
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
                             num_freqs: int = 10):
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
        num_freqs: int = 10,
        trace_dir:
    str = "/export2/kong102/energy_efficient_serving_results/request_timing"
    "/2025-04-28_lat-model-profiling/a40_qps9_reqs20000_fixed1740",
        log_dir_base:
    str = "/export2/obasit/EnergyEfficientServing/energy_efficient_serving_"
    "results/azure_trace_sampling/a40_qps9_reqs20000_fixed1740/",
        batch_type: Optional[str] = None):

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

            if (batch_type == "hybrid" and
                (len(prefill_lens) == 0 or len(decode_lens) == 0)
                    or batch_type == "prefill-only" and len(decode_lens) > 0
                    or batch_type == "decode-only" and len(prefill_lens) > 0):

                continue

            freq = test_freqs[count % len(test_freqs)]
            params.append(
                BenchmarkBatchParam(
                    prefill_input_lens=prefill_lens,
                    prefill_completed_input_lens=prefill_computed_lens,
                    decode_input_lens=decode_lens,
                    log_dir=f"{log_dir_base}/logs/batch_{count}_freq_{freq}",
                    gpu_freq_mhz=freq,
                    min_num_iters=2,
                    min_seconds=1,
                ))
    return params


def gen_compare_w_wo_precompute(
    tp: int,
    pp: int,
    num_freqs: int = 10,
):
    # test_freqs = uniform_sample_sorted(nvml_get_available_freq(), num_freqs)
    test_freqs = [825, 975, 1125, 1275]
    params = []

    prefill_input_lens = [29, 941]
    prefill_completed_input_lens = [0, 313]
    decode_input_lens = [
        1255, 1468, 854, 709, 932, 663, 1427, 582, 1185, 1591, 556, 574, 904,
        1248, 561, 616, 537, 517, 508, 485, 524, 974, 494, 481, 486, 507, 476,
        718, 462, 550, 418, 474, 461, 889, 426, 448, 480, 434, 546, 893, 434,
        884, 432, 857, 441, 365, 381, 364, 585, 460, 353, 413, 556, 356, 812,
        344, 588, 337, 386, 332, 340, 345, 952, 416, 1252, 373, 354, 629, 918,
        329, 391, 711, 306, 353, 320, 329, 314, 316, 732, 316, 310, 300, 1148,
        305, 315, 328, 1072, 385, 335, 322, 332, 305, 326, 481, 297, 273, 313,
        312, 1054, 281, 281, 353, 520, 270, 285, 290, 290, 262, 385, 261, 408,
        267, 276, 273, 260, 268, 263, 265, 294, 265, 251, 569, 247, 242, 241,
        233, 864, 244, 236, 474, 240, 566, 285, 235, 245, 863, 223, 220, 254,
        220, 331, 307, 212, 560, 217, 914, 480, 1069, 210, 226, 994, 208, 235,
        256, 191, 286, 186, 192, 328, 268, 1149, 187, 191, 209, 169, 175, 224,
        160, 165, 169, 243, 226, 255, 152, 146, 1204, 512, 142, 142, 139, 154,
        160, 145, 129, 135, 160, 129, 182, 153, 154, 565, 290, 122, 185, 579,
        131, 210, 140, 123, 127, 998, 137, 999, 552, 120, 120, 361, 988, 130,
        205, 106, 203, 883, 177, 189, 97, 130, 102, 108, 109, 101, 101, 149,
        891, 425, 124, 213, 123, 92, 89, 83, 86, 78, 697, 505, 91, 329, 158,
        109, 80, 496, 78, 91, 83, 352, 91, 119, 122, 78, 304, 335, 95, 76, 94,
        89, 72, 585, 132, 123, 125, 206, 65, 71, 77, 429, 61, 345, 167, 290,
        87, 83, 52, 173, 61, 129, 262, 732, 35, 67, 213, 71, 448, 52, 38, 823,
        28, 93, 147, 42, 53, 626, 588, 31, 106, 32, 112, 634, 35, 37, 61, 430,
        23, 20, 319, 29, 238, 16, 576, 770, 341, 347, 765
    ]
    decode_input_lens = [l + 1 for l in decode_input_lens]  # noqa

    for freq in test_freqs:
        p_w = BenchmarkBatchParam(
            prefill_input_lens=prefill_input_lens,
            prefill_completed_input_lens=prefill_completed_input_lens,
            decode_input_lens=decode_input_lens,
            gpu_freq_mhz=freq,
            min_num_iters=10,
            min_seconds=1,
            log_dir=f'./logs/freq{freq}_w_precompute',
        )
        p_wo = copy.deepcopy(p_w)
        p_wo.prefill_completed_input_lens = [
            0 for _ in range(len(p_wo.prefill_input_lens))
        ]
        p_wo.log_dir = f'./logs/freq{freq}_wo_precompute'
        params.extend([p_w, p_wo])
    return params


def main(expr_fn: Callable, model: str):
    tp = 1
    pp = 1
    vllm_args = (f"--model {model} "
                 f"-tp {tp} "
                 f"-pp {pp} "
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
    params = expr_fn(tp=tp, pp=pp)

    uvloop.run(benchmark_batch(vllm_args, params))


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
        'compare_w_wo_precompute': gen_compare_w_wo_precompute,
    }[sys.argv[1]]
    model = sys.argv[2]
    main(expr_fn, model)
