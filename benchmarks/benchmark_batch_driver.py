from pathlib import Path

import torch
import uvloop
from benchmark_batch import main as benchmark_batch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser


def get_gpu_name():
    return torch.cuda.get_device_name().split(' ')[-1]


def yield_benchmark_batch_args():
    prefill_input_len = 512
    decode_input_len = 512
    tp = 1
    pp = 1
    expr_dir = Path(
        '/export2/kong102/energy_efficient_serving_results/request_timing/2025-01-22_benchmark-batch'
    )

    for prefill_bs in [0, 1, 2, 4]:
        for decode_bs in [0, 1, 8, 64, 512, 2048, 8192]:

            if prefill_bs == 0 and decode_bs == 0:
                continue

            log_dir = expr_dir / \
                f'{get_gpu_name()}_tp{tp}_pp{pp}_prefill-len-{prefill_input_len}-bs-{prefill_bs}_decode-len-{decode_input_len}-bs-{decode_bs}'
            vllm_args = ("--model meta-llama/Llama-3.1-8B-Instruct "
                         f"-tp {tp} "
                         f"-pp {pp} "
                         "--collect-detailed-traces worker "
                         f"--log-dir {log_dir} ").split()

            parser = FlexibleArgumentParser(description="Benchmark per-batch.")
            parser = AsyncEngineArgs.add_cli_args(parser)
            vllm_args = parser.parse_args(vllm_args)

            yield {
                'args': vllm_args,
                'prefill_input_len': prefill_input_len,
                'prefill_bs': prefill_bs,
                'decode_input_len': decode_input_len,
                'decode_bs': decode_bs,
            }


def main():
    for benchmark_batch_args in yield_benchmark_batch_args():
        print(benchmark_batch_args['args'].log_dir)
        uvloop.run(benchmark_batch(**benchmark_batch_args))


if __name__ == '__main__':
    main()
