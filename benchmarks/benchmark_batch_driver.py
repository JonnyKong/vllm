# SPDX-License-Identifier: Apache-2.0
import itertools
import os

import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch
from benchmark_utils import (get_gpu_name, get_result_root,
                             uniform_sample_sorted)

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import nvml_get_available_freq
from vllm.utils import FlexibleArgumentParser


def yield_benchmark_batch_args(pp: int = 1,
                               tp: int = 1,
                               skip_existing: bool = False):
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


def yield_benchmark_idle_power_args(pp: int = 1, tp: int = 1):
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


def main():
    tp = 1
    pp = 1
    # model = 'meta-llama/Llama-3.1-8B-Instruct'
    model = 'meta-llama/Llama-3.1-70B-Instruct'
    vllm_args = (f"--model {model} "
                 f"-tp {tp} "
                 f"-pp {pp} "
                 "--collect-detailed-traces worker").split()
    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args)

    # Pass in a list instead of generator so tqdm prints progress
    uvloop.run(
        benchmark_batch(vllm_args,
                        list(yield_benchmark_batch_args(pp=pp, tp=tp))))
    # uvloop.run(
    #     benchmark_batch(vllm_args, list(yield_benchmark_idle_power_args())))


if __name__ == '__main__':
    main()
