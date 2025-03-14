# SPDX-License-Identifier: Apache-2.0
import random
from pathlib import Path

from benchmark_batch import BenchmarkBatchParam
from benchmark_utils import uniform_sample_sorted
from matplotlib import pyplot as plt
from scipy.stats import lognorm

from vllm.platforms import current_platform
from vllm.platforms.nvml_utils import nvml_get_available_freq

################ Knobs ###############
# Profiled empirically on A40, llama3-8b
MAX_DECODE_BS = 256
MAX_PREFILL_BS = 16
MIN_INPUT_LEN = 10
MAX_INPUT_LEN = 2048
######################################


def gen_benchmark_batch_args_sample_hybrid(num_samples: int, num_freqs: int):
    return _sample(num_samples=num_samples,
                   num_freqs=num_freqs,
                   enable_prefill=True,
                   enable_decode=True)


def gen_benchmark_batch_args_sample_prefill_only(num_samples: int,
                                                 num_freqs: int):
    return _sample(num_samples=num_samples,
                   num_freqs=num_freqs,
                   enable_prefill=True,
                   enable_decode=False)


def gen_benchmark_batch_args_sample_decode_only(num_samples: int,
                                                num_freqs: int):
    return _sample(num_samples=num_samples,
                   num_freqs=num_freqs,
                   enable_prefill=False,
                   enable_decode=True)


def _get_lognorm_generator(shape: float,
                           scale: float,
                           min_val: int,
                           max_val: int,
                           batch_size=100):
    """
    Fitted on ShareGPT. Uses batch sampling for improved performance.
    """
    batch = iter([])  # Empty iterator to start
    loc = 0.0

    while True:
        try:
            ret = int(round(next(batch)))
            ret = max(min_val, ret)
            ret = min(max_val, ret)
            yield ret
        except StopIteration:
            batch = iter(lognorm.rvs(shape, loc, scale,
                                     size=batch_size))  # Refill batch


def _get_req_len_generator(dataset: str, mode: str):
    assert dataset in ['sharegpt']
    assert mode in ['prefill', 'decode']
    lognorm_shape_scale_dict = {
        'sharegpt': {
            # [shape, scale]
            'prefill': [1.5502, 87.9664],
            'decode': [0.9236, 250.5623],
        },
    }
    param = lognorm_shape_scale_dict[dataset][mode]
    return _get_lognorm_generator(shape=param[0],
                                  scale=param[1],
                                  min_val=MIN_INPUT_LEN,
                                  max_val=MAX_INPUT_LEN)


def _sample(num_samples: int,
            num_freqs: int,
            enable_prefill: bool,
            enable_decode: bool,
            token_budget: int = 8192,
            seq_budget: int = 2048,
            seed: int = 0):
    current_platform.seed_everything(seed)
    assert enable_prefill or enable_decode

    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), num_freqs)
    prefill_len_gen = _get_req_len_generator('sharegpt', 'prefill')
    decode_len_gen = _get_req_len_generator('sharegpt', 'decode')

    def gen_one():
        token_budget_remaining = token_budget
        seq_budget_remaining = seq_budget

        # Decode
        decode_lens = []
        if enable_decode:
            decode_bs = random.randint(1, MAX_DECODE_BS)
            for _ in range(decode_bs):
                decode_lens.append(next(decode_len_gen))
                token_budget_remaining -= 1
                seq_budget_remaining -= 1
                if token_budget_remaining <= 0 or seq_budget_remaining <= 0:
                    break

        # Prefill
        prefill_lens = []
        if enable_prefill:
            prefill_bs = random.randint(1, MAX_PREFILL_BS)
            for _ in range(prefill_bs):
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

    return list(gen_one() for _ in range(num_samples))


def get_cdf_data(raw_data, scale=1.0):
    index = 1
    x_data = []
    y_data = []
    sorted_data = sorted(raw_data)
    for row in sorted_data:
        x_data.append(1.0 * row / scale)
        y_data.append(index * 1.0 / len(sorted_data) * 100)
        index += 1
    return [x_data, y_data]


if __name__ == '__main__':
    for fn in [
            gen_benchmark_batch_args_sample_hybrid,
            gen_benchmark_batch_args_sample_prefill_only,
            gen_benchmark_batch_args_sample_decode_only,
    ]:
        params = list(fn(num_samples=2000, num_freqs=11))
        dists_to_plot = {
            'prefill_len':
            [length for p in params for length in p.prefill_input_lens],
            'decode_len':
            [length for p in params for length in p.decode_input_lens],
            'prefill_bs': [len(p.prefill_input_lens) for p in params],
            'decode_bs': [len(p.decode_input_lens) for p in params],
        }

        fig, axs = plt.subplots(1, len(dists_to_plot), figsize=(15, 5))
        for i, (title, data) in enumerate(dists_to_plot.items()):
            ax = axs[i]
            x, y = get_cdf_data(data)
            ax.plot(x, y)
            ax.set_title(title)
        fig.tight_layout()
        Path('figs').mkdir(exist_ok=True)
        plt.savefig(Path('figs') / f'{fn.__name__}.pdf')
        plt.close(fig)
