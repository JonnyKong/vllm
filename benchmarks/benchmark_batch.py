import argparse
import asyncio
import os
import random
from typing import Dict, List

import tqdm
import uvloop
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm import SamplingParams
from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import PerfMetricCSVLogger
from vllm.engine.metrics_types import Stats
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (ExecuteModelRequest, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import FlexibleArgumentParser, cdiv

request_id = 0


async def main(
    args: argparse.Namespace,
    prefill_input_lens: List[int],
    decode_input_lens: List[int],
    disable_frontend_multiprocessing: bool = True,
    num_iters: int = 100,
):
    """
    Feed executor with ExecuteModelRequest similar to how it's done in
    `AsyncLLMEngine`
    """
    random.seed(args.seed)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    assert disable_frontend_multiprocessing, \
        '''
            setting disable_frontend_multiprocessing=True will use
            MQLLMEngineClient instead of AsyncLLMEngine, which is not supported
            for now'
        '''

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)

    # The `PerfMetricCSVLogger` of `LLMEngine` will not be invoked when we
    # directly call the executor, so we create another logger outside it
    perf_metric_logger = PerfMetricCSVLogger(
        filename=f"{args.log_dir}/perf_metric_{os.getpid()}.csv")

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        assert isinstance(llm, AsyncLLMEngine)

        executor = llm.engine.model_executor
        pipeline_parallel_size \
                = llm.engine.parallel_config.pipeline_parallel_size

        # Keep `pipeline_parallel_size` instances of `execute_model_async()`
        # running concurrently
        initial_requests = [
            build_dummy_execute_model_request(
                llm,
                tokenizer,
                prefill_input_lens=prefill_input_lens,
                decode_input_lens=decode_input_lens)
            for ve in range(pipeline_parallel_size)
        ]
        requests_in_progress = [
            asyncio.create_task(executor.execute_model_async(req))
            for req in initial_requests
        ]

        for iter in tqdm.tqdm(range(num_iters)):
            done, _ = await asyncio.wait(requests_in_progress,
                                         return_when=asyncio.FIRST_COMPLETED)
            for _ in range(pipeline_parallel_size):
                await asyncio.sleep(0)
            for task in done:
                output = task.result()
                perf_metric_logger.log(get_stats(llm, output))

                # Insert new req
                virtual_engine = requests_in_progress.index(task)
                req = build_dummy_execute_model_request(
                    llm,
                    tokenizer,
                    prefill_input_lens=prefill_input_lens,
                    decode_input_lens=decode_input_lens)
                requests_in_progress[virtual_engine] = asyncio.create_task(
                    executor.execute_model_async(req))

        # Cleanup
        _ = await asyncio.wait(requests_in_progress,
                               return_when=asyncio.ALL_COMPLETED)

    perf_metric_logger.persist_to_disk()


def build_dummy_execute_model_request(
    llm: AsyncLLMEngine,
    tokenizer: PreTrainedTokenizerBase,
    prefill_input_lens: List[int],
    decode_input_lens: List[int],
) -> ExecuteModelRequest:
    seq_group_metadata_list: List[SequenceGroupMetadata] = \
        [build_dummy_seq_group_metadata(
            llm, tokenizer, input_len, is_prompt=True)
         for input_len in prefill_input_lens] + \
        [build_dummy_seq_group_metadata(
            llm, tokenizer, input_len, is_prompt=False)
         for input_len in decode_input_lens]
    return ExecuteModelRequest(seq_group_metadata_list=seq_group_metadata_list,
                               # All the rest stay as default
                               )


def build_dummy_seq_group_metadata(
    llm: AsyncLLMEngine,
    tokenizer: PreTrainedTokenizerBase,
    input_len: int,
    is_prompt: bool,
) -> SequenceGroupMetadata:
    """
    Send requests as new every time (no `SequenceGroupMetadataDelta`).
    """
    global request_id

    # Each `SequenceGroup` contains one random seq
    seq_data: Dict[int, SequenceData] = {
        0:
        SequenceData.from_seqs([
            random.randint(0, tokenizer.vocab_size - 1)
            for _ in range(input_len)
        ])
    }

    # Same as in `benchmark_throughput.py`
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=2048,  # TODO: remove this hardcoded value
    )

    # Build a random block mapping
    # TODO: try sequential block tables
    block_manager = llm.engine.scheduler[0].block_manager
    assert isinstance(block_manager, SelfAttnBlockSpaceManager)
    block_size = block_manager.block_size
    num_required_blocks = cdiv(input_len, block_size)
    block_tables: Dict[int, List[int]] = {
        0: [
            random.randint(0, block_manager.num_total_gpu_blocks)
            for _ in range(num_required_blocks)
        ]
    }

    # For simplicity, assume all prefill and decode requires sampling. In
    # practice, if prefill is chunked, only the last chunk requires sampling
    do_sample = True

    ret = SequenceGroupMetadata(
        request_id=str(request_id),
        is_prompt=is_prompt,
        seq_data=seq_data,
        sampling_params=sampling_params,
        block_tables=block_tables,
        do_sample=do_sample,
        # Assume the rest doesn't matter and uses defaults
    )
    request_id += 1
    return ret


def get_stats(llm: AsyncLLMEngine, model_output: List[SamplerOutput]) -> Stats:
    return llm.engine._get_stats(
        scheduler_outputs=None,
        model_output=model_output,
        finished_before=None,
        skip=None,
    )


if __name__ == '__main__':
    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    uvloop.run(
        main(args,
             prefill_input_lens=[1024, 1024],
             decode_input_lens=[100, 200, 300, 400, 500]))
