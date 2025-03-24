# SPDX-License-Identifier: Apache-2.0
import contextlib
import gc
import random

import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)


@contextlib.contextmanager
def disable_python_gc():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
            gc.collect()


if __name__ == '__main__':
    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = ("--model meta-llama/Llama-3.1-8B-Instruct "
                 "-tp 1 -pp 1 "
                 "--max-num-seqs 1024 --max-num-batched-tokens 8192 "
                 "--max-model-len 65536 "
                 "--collect-detailed-traces worker "
                 "--disable-async-output-pro").split()
    vllm_args = parser.parse_args(vllm_args)

    benchmark_batch_param = []
    test_freqs = [210, 360, 510, 675, 825, 975, 1125, 1275, 1440, 1590, 1740]
    test_batches = [i for i in range(4, 450, 4)]
    for batch_size in test_batches:
        for freq in test_freqs:
            decode_input_lens = [
                int(random.gauss(200, 50)) for _ in range(batch_size)
            ]
            batch_log_dir = f'{vllm_args.log_dir}/{batch_size}_batch_ \
                            {str(freq)}_{sum(decode_input_lens)}_total_tokens'

            benchmark_batch_param_i = BenchmarkBatchParam(
                prefill_input_lens=[],
                decode_input_lens=decode_input_lens,
                log_dir=batch_log_dir,
                gpu_freq_mhz=freq,
                gpu_power_meas_interval=0.001,
                delay_time_min_s=0,
                delay_time_max_s=0,
                min_num_iters=100,
                min_seconds=0,
            )
            benchmark_batch_param.append(benchmark_batch_param_i)
    uvloop.run(benchmark_batch(vllm_args, benchmark_batch_param))
