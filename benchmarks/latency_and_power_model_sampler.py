# SPDX-License-Identifier: Apache-2.0
import random

import pandas as pd
from benchmark_batch import BenchmarkBatchParam
from benchmark_utils import uniform_sample_sorted
from scipy.stats import lognorm

from vllm.platforms.nvml_utils import nvml_get_available_freq


def yield_benchmark_batch_args_sample_hybrid(num_samples: int, num_freqs: int):
    yield from _sample(num_samples=num_samples,
                       num_freqs=num_freqs,
                       enable_prefill=True,
                       enable_decode=True)


def yield_benchmark_batch_args_sample_prefill_only(num_samples: int,
                                                   num_freqs: int):
    yield from _sample(num_samples=num_samples,
                       num_freqs=num_freqs,
                       enable_prefill=True,
                       enable_decode=False)


def yield_benchmark_batch_args_sample_decode_only(num_samples: int,
                                                  num_freqs: int):
    yield from _sample(num_samples=num_samples,
                       num_freqs=num_freqs,
                       enable_prefill=False,
                       enable_decode=True)


def _get_prefill_len_generator(batch_size=100):
    """
    Fitted on ShareGPT. Uses batch sampling for improved performance.
    """
    shape = 1.5502
    loc = 0.0
    scale = 87.9664
    min_value = 10
    batch = iter([])  # Empty iterator to start

    while True:
        try:
            yield max(min_value, int(round(next(batch))))
        except StopIteration:
            batch = iter(lognorm.rvs(shape, loc, scale,
                                     size=batch_size))  # Refill batch


def _get_decode_len_generator(batch_size=100):
    """
    Fitted on ShareGPT. Uses batch sampling for improved performance.
    """
    shape = 0.8552
    loc = 0.0
    scale = 234.9029
    min_value = 10
    batch = iter([])  # Empty iterator to start

    while True:
        try:
            yield max(min_value, int(round(next(batch))))
        except StopIteration:
            batch = iter(lognorm.rvs(shape, loc, scale,
                                     size=batch_size))  # Refill batch


def _sample(num_samples: int,
            num_freqs: int,
            enable_prefill: bool = True,
            enable_decode: bool = True,
            token_budget: int = 8192,
            seq_budget: int = 2048):
    assert enable_prefill or enable_decode

    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), num_freqs)
    prefill_len_gen = _get_prefill_len_generator()
    decode_len_gen = _get_decode_len_generator()

    def gen_one():
        token_budget_remaining = token_budget
        seq_budget_remaining = seq_budget

        # Decode
        decode_lens = []
        if enable_decode:
            # Select a decode_bs up to seq_budget, otherwise seq_budget will
            # always be exhausted by decodes
            decode_bs = random.randint(1, seq_budget)
            for _ in range(decode_bs):
                decode_lens.append(next(decode_len_gen))
                token_budget_remaining -= 1
                seq_budget_remaining -= 1
                if token_budget_remaining <= 0 or seq_budget_remaining <= 0:
                    break

        # Prefill
        prefill_lens = []
        if enable_prefill:
            while True:
                length = next(prefill_len_gen)
                token_budget_remaining -= length
                seq_budget_remaining -= 1
                if token_budget_remaining <= 0 or seq_budget_remaining <= 0:
                    break
                prefill_lens.append(length)

        return BenchmarkBatchParam(
            prefill_input_lens=prefill_lens,
            decode_input_lens=decode_lens,
            log_dir="/tmp/results",
            gpu_freq_mhz=random.choice(test_freqs),
            min_num_iters=2,
            min_seconds=1,
        )

    yield from (gen_one() for _ in range(num_samples))


if __name__ == '__main__':
    for fn in [
            yield_benchmark_batch_args_sample_hybrid,
            yield_benchmark_batch_args_sample_prefill_only,
            yield_benchmark_batch_args_sample_decode_only,
    ]:
        params = list(fn(num_samples=2000, num_freqs=11))
        prefill_lens = pd.Series(
            [length for p in params for length in p.prefill_input_lens])
        decode_lens = pd.Series(
            [length for p in params for length in p.decode_input_lens])

        print(fn.__name__)
        print('prefill lens: ', prefill_lens.describe())
        print('decode lens: ', decode_lens.describe())
