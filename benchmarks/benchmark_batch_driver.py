import itertools
import os
from pathlib import Path

import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import nvml_get_available_freq
from vllm.utils import FlexibleArgumentParser


def uniform_sample_sorted(lst, k):
    """
    Selects `k` elements from the sorted input list as uniformly as possible,
    ensuring the first and last elements are included.
    """
    if k < 2 or k > len(lst):
        raise ValueError(
            "k must be at least 2 and at most the length of the list")
    lst = sorted(lst)
    step = (len(lst) - 1) / (k - 1)
    indices = sorted(set(round(i * step) for i in range(k)))
    return [lst[i] for i in indices]


def yield_benchmark_batch_args(skip_existing: bool = False):
    expr_dir = Path(
        '/export2/kong102/energy_efficient_serving_results/request_timing/2025-01-29_benchmark-batch/A40-pp1-tp1'
    )

    prefill_input_lens = [256, 1024]
    prefill_bss = [0, 1, 8]
    decode_input_lens = [256, 1024]
    decode_bss = [0, 8, 256]
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), 16)

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


def yield_benchmark_idle_power_args():
    """
    Introduce a delay before issuing each batch to assess whether the power
    consumption between batches aligns with the idle power measured offline.
    """
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), 16)
    prefill_input_len = 1024
    prefill_bs = 2
    decode_input_len = 1024
    decode_bs = 256

    for delay_time_s in [0.5, 2.0]:
        for freq in test_freqs:
            expr_dir = Path(
                f'/export2/kong102/energy_efficient_serving_results/request_timing/2025-02-02_benchmark-idle-power/A40-pp1-tp1-delay{delay_time_s}'
            )
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
    vllm_args = ("--model meta-llama/Llama-3.1-8B-Instruct "
                 f"-tp {tp} "
                 f"-pp {pp} "
                 "--collect-detailed-traces worker").split()
    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args)

    # Pass in a list instead of generator so tqdm prints progress
    # uvloop.run(benchmark_batch(vllm_args, list(yield_benchmark_batch_args())))
    uvloop.run(
        benchmark_batch(vllm_args, list(yield_benchmark_idle_power_args())))


if __name__ == '__main__':
    main()
