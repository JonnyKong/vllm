# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import uvloop
from benchmark_batch import BenchmarkBatchParam, benchmark_batch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

A40_FREQ_CHOICES = [
    540, 660, 780, 900, 1020, 1140, 1260, 1380, 1500, 1620, 1740
]


def main():
    model = 'meta-llama/Llama-3.1-8B-Instruct'
    vllm_args = (f"--model {model} "
                 "-tp 1 -pp 1 "
                 "--max-num-seqs 1024 --max-num-batched-tokens 8192 "
                 "--disable-async-output-proc "
                 "--collect-detailed-traces worker,power").split()
    result_root = Path(
        '/export2/kong102/energy_efficient_serving_results/request_timing/2025-03-06_power-model/a40_llama3-8b_benchmark-batch-for-power'
    )

    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args)

    prefill_input_lens = [128, 256, 512, 1024]
    decode_input_lens = list(range(4000, 4000 + 256))
    min_num_iters: int = 128
    min_seconds: int = 0

    params = []
    for freq in A40_FREQ_CHOICES:
        params.extend([
            BenchmarkBatchParam(
                prefill_input_lens=[],
                decode_input_lens=decode_input_lens,
                log_dir=str(result_root / f'decode-only-{freq}'),
                gpu_freq_mhz=freq,
                min_num_iters=min_num_iters,
                min_seconds=min_seconds,
            ),
            BenchmarkBatchParam(
                prefill_input_lens=prefill_input_lens,
                decode_input_lens=[],
                log_dir=str(result_root / f'prefill-only-{freq}'),
                gpu_freq_mhz=freq,
                min_num_iters=min_num_iters,
                min_seconds=min_seconds,
            ),
            BenchmarkBatchParam(
                prefill_input_lens=prefill_input_lens,
                decode_input_lens=decode_input_lens,
                log_dir=str(result_root / f'hybrid-only-{freq}'),
                gpu_freq_mhz=freq,
                min_num_iters=min_num_iters,
                min_seconds=min_seconds,
            ),
        ])

    uvloop.run(benchmark_batch(vllm_args, params))


if __name__ == '__main__':
    main()
