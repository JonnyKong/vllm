# SPDX-License-Identifier: Apache-2.0
import itertools
import json
import os
import random
from typing import List, Optional

import numpy as np
import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch
from benchmark_utils import (get_gpu_name, get_result_root,
                             uniform_sample_sorted)

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import nvml_get_available_freq
from vllm.utils import FlexibleArgumentParser


def yield_benchmark_batch_args_sample(pp: int = 1,
                                      tp: int = 1,
                                      num_samples: int = 10,
                                      num_freqs: int = 8,
                                      skip_existing: bool = False):

    max_prefill_input_len = 2048
    max_decode_input_len = 16384
    max_decode_bs = 512
    max_prefill_bs = 16
    test_freqs = uniform_sample_sorted(nvml_get_available_freq(), num_freqs)

    for j in range(num_samples):

        # Uncomment to choose random frequency
        #freq = random.choice(test_freqs)

        # Uncomment to choose uniform frequencies
        freq = test_freqs[j % len(test_freqs)]

        prefill_lens = []
        decode_lens = []

        # Uncomment to allow decodes in batch
        decode_bs = random.randint(1, max_decode_bs)

        # Uncomment if you want a prefill-only batch
        #decode_bs = 0

        # Chance of having prefill batch size be 1 with size 2048
        if (random.random() > 0.9):
            prefill_bs = 1
            prefill_lens.append(max_prefill_input_len)
        else:
            prefill_bs = random.randint(1, max_prefill_bs)

        # Fill in prefill lengths if not size 1
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

        if (decode_bs > 0):
            decode_lens = [0] * decode_bs
            bound_1 = random.randint(1, max_decode_input_len)
            bound_2 = random.randint(1, max_decode_input_len)
            decode_sample_max_length = max(bound_1, bound_2)
            decode_sample_min_length = min(bound_1, bound_2)
            for i in range(decode_bs):
                decode_lens[i] = random.randint(decode_sample_min_length,
                                                decode_sample_max_length)

        expr_dir = (
            get_result_root() /
            f'request_timing/2025-02-02_benchmark-batch_llama70b/{get_gpu_name()}-pp{pp}-tp{tp}'
        )
        log_dir = expr_dir / \
            f'prefill-sum-{sum(prefill_lens)}-max-{max(prefill_lens)}-bs-{prefill_bs}_decode-sum-{sum(decode_lens)}-max-{max(decode_lens)}-bs-{decode_bs}_freq-{freq}'
        if skip_existing and os.path.exists(log_dir):
            continue

        yield BenchmarkBatchParam(prefill_input_lens=prefill_lens,
                                  decode_input_lens=decode_lens,
                                  log_dir=str(log_dir),
                                  gpu_freq_mhz=freq,
                                  min_num_iters=1,
                                  min_seconds=1)


def yield_benchmark_batch_args_sample_decode_only(
    pp: int = 1,
    tp: int = 1,
    num_freqs: int = 8,
    num_bs: int = 8,
    decode_bounds: Optional[List] = None,
    skip_existing: bool = False,
    num_samples: Optional[int] = None,
):

    max_decode_bs = 512

    # The following 3 parameters can be changed according to
    # how many samples are desired.

    # num_freqs: Number of frequencies to test

    # num_bs: Number of decode batch sizes to test
    # Batch sizes tested will be evenly split up to
    # max_decode_bs

    # decode_bounds: Bounds to test in each configuration represented
    # as a list of tuples (lower bound, upper bound)
    # 16384 is a good upper bound for decode length, it is likely
    # not worth testing decode lengths higher than 16384.
    # The knobs in the default configuration tests bounds with range 16,
    # 256, 512, 2048, 4096, 8192, and 16384. These ranges were also
    # evenly spread up to max_decode_len to get samples across
    # various distributions and sample lengths.
    if decode_bounds is None:
        decode_bounds = [(1, 16), (4096, 4112), (8192, 8208), (12288, 12304),
                         (16368, 16384), (1, 256), (2048, 2560), (6144, 6656),
                         (10240, 10752), (14328, 14840), (16128, 16384),
                         (1, 2048), (1, 4096), (6144, 10240), (12288, 16384),
                         (14336, 16384), (1, 8192), (8192, 16384), (1, 16384),
                         (1, 16384)]

    if num_samples is None:
        num_samples = num_freqs * num_bs * len(decode_bounds)
    assert num_samples <= num_freqs * num_bs * len(
        decode_bounds
    ), "Number of samples cannot be greater than combinations of " + \
    "parameters given. Decrease num_samples or increase search space"

    freq_knob_arr = uniform_sample_sorted(nvml_get_available_freq(), num_freqs)
    decode_bs_knob_arr = [0] * num_bs
    decode_bs_step = max_decode_bs / num_bs

    for i in range(num_bs):
        decode_bs_knob_arr[i] = (int(
            (i + 0.25) * decode_bs_step), int((i + 1) * decode_bs_step))

    for freq_knob, decode_bs_knob, decode_bound_knob in \
            random.sample(sorted(itertools.product(
                freq_knob_arr,
                decode_bs_knob_arr,
                decode_bounds
            )), num_samples):

        freq = freq_knob
        decode_bs = random.randint(decode_bs_knob[0], decode_bs_knob[1])

        prefill_lens = []
        decode_lens = [0] * decode_bs

        # Fill in decode tokens
        for i in range(decode_bs):
            decode_lens[i] = random.randint(decode_bound_knob[0],
                                            decode_bound_knob[1])

        expr_dir = (
            get_result_root() /
            f'request_timing/2025-02-02_benchmark-batch_llama70b/{get_gpu_name()}-pp{pp}-tp{tp}'
        )
        log_dir = expr_dir / \
            f'decode-sum-{sum(decode_lens)}-max-{max(decode_lens)}-bs-{decode_bs}_freq-{freq}'
        if skip_existing and os.path.exists(log_dir):
            continue

        yield BenchmarkBatchParam(prefill_input_lens=prefill_lens,
                                  decode_input_lens=decode_lens,
                                  log_dir=str(log_dir),
                                  gpu_freq_mhz=freq,
                                  min_num_iters=1,
                                  min_seconds=1)


def main():
    tp = 1
    pp = 1
    model = 'meta-llama/Llama-3.1-8B-Instruct'

    # These two parameters affect the file the profile
    # gets saved into
    model_name = 'LLama3-8B'
    gpu = get_gpu_name()

    vllm_args = (f"--model {model} "
                 f"-tp {tp} "
                 f"-pp {pp} "
                 "--collect-detailed-traces worker").split()
    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args)
    vllm_args.max_model_len = 10000

    # Pass in a list instead of generator so tqdm prints progress
    try:
        with open(f'profilers/profile_{gpu}_{model_name}.json') as f:
            profile = json.load(f)
    except FileNotFoundError:
        profile = []

    print(len(profile), "samples")
    latencies = []

    benchmark_param_args = list(
        yield_benchmark_batch_args_sample_decode_only(pp=pp,
                                                      tp=tp,
                                                      num_samples=10))
    try:
        uvloop.run(benchmark_batch(vllm_args, benchmark_param_args, latencies))
    except Exception:
        print("Something went wrong at sample", len(latencies))
    except KeyboardInterrupt:
        print("Stopped collecting samples at sample", len(latencies))

    for i, lat in enumerate(latencies):
        profile_data = dict()
        profile_data["latencies"] = lat
        profile_data["frequency"] = benchmark_param_args[i].gpu_freq_mhz
        profile_data["prefill_batch_size"] = len(
            benchmark_param_args[i].prefill_input_lens)
        profile_data["decode_batch_size"] = len(
            benchmark_param_args[i].decode_input_lens)
        profile_data["sum_prefill_len"] = sum(
            benchmark_param_args[i].prefill_input_lens)
        profile_data["sum_decode_len"] = sum(
            benchmark_param_args[i].decode_input_lens)
        profile_data["max_prefill_len"] = 0 if profile_data[
            "prefill_batch_size"] == 0 else max(
                benchmark_param_args[i].prefill_input_lens)
        profile_data["max_decode_len"] = 0 if profile_data[
            "decode_batch_size"] == 0 else max(
                benchmark_param_args[i].decode_input_lens)
        profile_data["std_prefill_len"] = 0 if profile_data[
            "prefill_batch_size"] == 0 else np.std(
                benchmark_param_args[i].prefill_input_lens)
        profile_data["std_decode_len"] = 0 if profile_data[
            "decode_batch_size"] == 0 else np.std(
                benchmark_param_args[i].decode_input_lens)

        profile.append(profile_data)

    print("Dumping", len(profile), "samples")
    with open(f'profilers/profile_{gpu}_{model_name}.json', "w") as f:
        json.dump(profile, f, indent=4)


if __name__ == '__main__':
    main()
