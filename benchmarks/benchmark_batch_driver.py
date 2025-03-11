# SPDX-License-Identifier: Apache-2.0
import itertools
import os
import random
import sys
from typing import Callable

import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch
from benchmark_utils import (get_gpu_name, get_result_root,
                             uniform_sample_sorted)

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import nvml_get_available_freq
from vllm.utils import FlexibleArgumentParser


def yield_benchmark_batch_args(pp: int, tp: int, skip_existing: bool = False):
    expr_dir = (
        get_result_root() /
        f'request_timing/2025-02-02_benchmark-batch_llama70b/{get_gpu_name()}-pp{pp}-tp{tp}'
    )

    prefill_input_lens = [2048]
    prefill_bss = [1]
    decode_input_lens = [1024, 16384]
    decode_bss = [256]
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), 8)

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

        yield BenchmarkBatchParam(
            prefill_input_lens=[prefill_input_len] * prefill_bs,
            decode_input_lens=[decode_input_len] * decode_bs,
            log_dir=str(log_dir),
            gpu_freq_mhz=freq,
        )


def yield_benchmark_idle_power_args(pp: int, tp: int):
    """
    Introduce a delay before issuing each batch to assess whether the power
    consumption between batches aligns with the idle power measured offline.
    """
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), 16)
    prefill_input_len = 1024
    prefill_bs = 2
    decode_input_len = 1024
    decode_bs = 256

    for delay_time_s in [2.0]:
        for freq in test_freqs:
            expr_dir = (get_result_root() / 'request_timing' /
                        '2025-02-02_benchmark-idle-power' /
                        f'{get_gpu_name()}-pp{pp}-tp{tp}-delay{delay_time_s}')
            log_dir = expr_dir / \
                f'prefill-len-{prefill_input_len}-bs-{prefill_bs}_decode-len-{decode_input_len}-bs-{decode_bs}_freq-{freq}'
            yield BenchmarkBatchParam(
                prefill_input_lens=[prefill_input_len] * prefill_bs,
                decode_input_lens=[decode_input_len] * decode_bs,
                log_dir=str(log_dir),
                gpu_freq_mhz=freq,
                delay_time_s=delay_time_s,
            )


def yield_benchmark_batch_args_sample(pp: int = 1,
                                      tp: int = 1,
                                      skip_existing: bool = False,
                                      num_samples: int = 10,
                                      num_freqs: int = 11):
    """
    TODO: remove this function with the one in `latency_profiler.py` once that
    becomes stable.
    """
    max_prefill_input_len = 2048
    max_decode_input_len = 16384
    max_decode_bs = 512
    max_prefill_bs = 16
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), num_freqs)

    for j in range(num_samples):

        freq = test_freqs[j % len(test_freqs)]
        expr_dir = get_result_root()
        # Choose random frequency
        #freq = random.choice(test_freqs)
        prefill_lens = []
        decode_lens = []
        # Have some samples be prefill-only or decode-only
        if (random.random() > 0.00):
            prefill_bs = 0
        else:
            if (random.random() > 0.9):
                prefill_bs = 1
                prefill_lens.append(max_prefill_input_len)
            else:
                prefill_bs = random.randint(1, max_prefill_bs)
        if (random.random() > 1.00):
            decode_bs = 0
        else:
            decode_bs = random.randint(1, max_decode_bs)
        if prefill_bs == 0 and decode_bs == 0:
            decode_bs = random.randint(1, max_decode_bs)
        decode_bs = random.randint(1, max_decode_bs)

        if (random.random() > 0.9):
            prefill_bs = 1
            prefill_lens.append(max_prefill_input_len)
        else:
            prefill_bs = random.randint(1, max_prefill_bs)
        # Fill in prefill lengths
        if (len(prefill_lens) == 0):
            retry = 0
            if (random.random() > 0.5):
                prefill_sample_token_limit = random.randint(
                    1, max_prefill_input_len)
            else:
                prefill_sample_token_limit = max_prefill_input_len
            while (retry < 3
                   and prefill_sample_token_limit < max_prefill_input_len / 2):
                prefill_sample_token_limit = random.randint(
                    1, max_prefill_input_len)
                retry += 1
            prefill_length = 0
            num_prefills = 0
            retry = 0
            prefill_sample_len_limit = random.randint(
                1, prefill_sample_token_limit)
            while (prefill_length < prefill_sample_token_limit
                   and num_prefills < prefill_bs and retry < 3):
                next_prefill_length = random.randint(1,
                                                     prefill_sample_len_limit)
                if (prefill_length + next_prefill_length
                        <= prefill_sample_token_limit):
                    prefill_lens.append(next_prefill_length)
                    num_prefills += 1
                    prefill_length += next_prefill_length
                    retry = 0
                else:
                    retry += 1
            prefill_bs = num_prefills

        # Fill in decode tokens
        decode_lens = [0] * decode_bs
        bound_1 = random.randint(1, max_decode_input_len)
        bound_2 = random.randint(1, max_decode_input_len)
        decode_sample_max_length = max(bound_1, bound_2)
        decode_sample_min_length = min(bound_1, bound_2)
        for i in range(decode_bs):
            decode_lens[i] = random.randint(decode_sample_min_length,
                                            decode_sample_max_length)

        log_dir = expr_dir / \
            'results'
        if skip_existing and os.path.exists(log_dir):
            continue

        yield BenchmarkBatchParam(prefill_input_lens=prefill_lens,
                                  decode_input_lens=decode_lens,
                                  log_dir=str(log_dir),
                                  gpu_freq_mhz=freq,
                                  min_num_iters=1,
                                  min_seconds=1)


def yield_benchmark_sarathi_args(pp: int, tp: int):
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
    yield BenchmarkBatchParam(
        prefill_input_lens=[],
        decode_input_lens=decode_input_lens,
        log_dir=str(log_dir),
        gpu_freq_mhz=freq,
        delay_time_min_s=0.0,
        delay_time_max_s=0.0,
    )


def yield_benchmark_power_profiling(pp: int, tp: int):
    expr_dir = (
        get_result_root() /
        f'request_timing/2025-03-10_power-model-profiling/{get_gpu_name()}-pp{pp}-tp{tp}_llama8-3b'
    )

    random.seed(0)
    for i, args in enumerate(
            yield_benchmark_batch_args_sample(num_samples=6000,
                                              num_freqs=11,
                                              skip_existing=True)):
        args.log_dir = str(expr_dir / f'sample{i}_freq{args.gpu_freq_mhz}')
        yield args


def main(expr_fn: Callable):
    tp = 1
    pp = 1
    model = 'meta-llama/Llama-3.1-8B-Instruct'
    # model = 'meta-llama/Llama-3.1-70B-Instruct'
    vllm_args = (f"--model {model} "
                 f"-tp {tp} "
                 f"-pp {pp} "
                 "--disable-async-output-proc "
                 "--max-num-seqs 1024 --max-num-batched-tokens 8192 "
                 "--collect-detailed-traces worker").split()
    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args)

    # Pass in a list instead of generator so tqdm prints progress
    uvloop.run(benchmark_batch(vllm_args, list(expr_fn(tp=tp, pp=pp))))


if __name__ == '__main__':
    expr_fn = {
        'batch': yield_benchmark_batch_args,
        'idle-power': yield_benchmark_idle_power_args,
        'sarathi-serve-sla': yield_benchmark_sarathi_args,
        'power_profiling': yield_benchmark_power_profiling,
    }[sys.argv[1]]
    main(expr_fn)
