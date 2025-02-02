import argparse
import asyncio
import contextlib
import gc
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
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
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms.nvml_power_monitor import measure_power
from vllm.platforms.nvml_utils import nvml_get_available_freq, nvml_set_freq
from vllm.sequence import (ExecuteModelRequest, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import FlexibleArgumentParser, cdiv, random_uuid

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


@contextlib.contextmanager
def log_perf_metric(filename: str):
    perf_logger = None
    try:
        perf_logger = PerfMetricCSVLogger(
            filename=filename, disable_periodic_persist_to_disk=True)
        yield perf_logger
    finally:
        if perf_logger:
            perf_logger.persist_to_disk()


def cyclic_generator(lst: Iterable):
    while True:
        yield from lst


@dataclass
class BenchmarkBatchParam:
    prefill_input_lens: List[int]
    decode_input_lens: List[int]
    log_dir: str
    gpu_freq_mhz: int
    delay_time_s: float = 0.0  # Delay before issuing each batch.

    # Run terminates when both reaches
    min_num_iters: int = 10
    min_seconds: int = 5


async def benchmark_batch(
    vllm_args: argparse.Namespace,
    params: Iterable[BenchmarkBatchParam],
):
    """
    Feed executor with ExecuteModelRequest similar to how it's done in
    `AsyncLLMEngine`
    """
    random.seed(vllm_args.seed)

    engine_args = AsyncEngineArgs.from_cli_args(vllm_args)
    disable_frontend_multiprocessing = True
    assert disable_frontend_multiprocessing, \
        '''
            setting disable_frontend_multiprocessing=True will use
            MQLLMEngineClient instead of AsyncLLMEngine, which is not supported
            for now'
        '''

    tokenizer = AutoTokenizer.from_pretrained(
        vllm_args.model, trust_remote_code=vllm_args.trust_remote_code)

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        assert isinstance(llm, AsyncLLMEngine)

        executor = llm.engine.model_executor
        pipeline_parallel_size \
                = llm.engine.parallel_config.pipeline_parallel_size

        # Keep `pipeline_parallel_size` instances of `execute_model_async()`
        # running concurrently
        for param in tqdm.tqdm(params):
            # Construct requests eagarly so request creation does not block the
            # critical path. Create more than `param.min_num_iters` requests to
            # prevent wrap around and send same request multiple times and
            # affecting the cache hit rate
            requests = [
                build_dummy_execute_model_request(llm, tokenizer, param)
                for _ in range(param.min_num_iters * 4)
            ]
            request_gen = cyclic_generator(requests)

            initial_requests = [
                next(request_gen) for ve in range(pipeline_parallel_size)
            ]
            requests_in_progress = [
                asyncio.create_task(executor.execute_model_async(req))
                for req in initial_requests
            ]

            # The `PerfMetricCSVLogger` of `LLMEngine` will not be invoked when
            # we directly call the executor, so we create another logger
            # outside of it
            energy_log = os.path.join(param.log_dir, 'power_log.csv')
            perf_log = os.path.join(param.log_dir, 'perf_metric.csv')
            with disable_python_gc(), \
                    measure_power(energy_log), \
                    log_perf_metric(perf_log) as perf_metric_logger, \
                    nvml_set_freq(param.gpu_freq_mhz):
                time_start = time.perf_counter()
                iter = 0
                while True:
                    done, _ = await asyncio.wait(
                        requests_in_progress,
                        return_when=asyncio.FIRST_COMPLETED)
                    for _ in range(pipeline_parallel_size):
                        await asyncio.sleep(0)
                    for task in done:
                        output = task.result()
                        perf_metric_logger.log(get_stats(llm, output))

                        # Insert new req
                        virtual_engine = requests_in_progress.index(task)
                        req = next(request_gen)
                        await asyncio.sleep(param.delay_time_s)
                        requests_in_progress[
                            virtual_engine] = asyncio.create_task(
                                executor.execute_model_async(req))

                    iter += 1
                    if (iter >= param.min_num_iters
                            and time.perf_counter() - time_start >
                            param.min_seconds):
                        logger.info(
                            'Run terminated on %d iters and %d seconds',
                            param.min_num_iters, param.min_seconds)
                        break

                # Cleanup
                _ = await asyncio.wait(requests_in_progress,
                                       return_when=asyncio.ALL_COMPLETED)


def build_dummy_execute_model_request(
        llm: AsyncLLMEngine, tokenizer: PreTrainedTokenizerBase,
        benchmark_batch_param: BenchmarkBatchParam):
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    for input_len in benchmark_batch_param.prefill_input_lens:
        seq_group_metadata_list.append(
            build_dummy_seq_group_metadata(llm,
                                           tokenizer,
                                           input_len,
                                           is_prompt=True))
    for input_len in benchmark_batch_param.decode_input_lens:
        seq_group_metadata_list.append(
            build_dummy_seq_group_metadata(llm,
                                           tokenizer,
                                           input_len,
                                           is_prompt=False))
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
    seq = SequenceData.from_seqs([
        random.randint(0, tokenizer.vocab_size - 1) for _ in range(input_len)
    ])
    if not is_prompt:
        seq.update_num_computed_tokens(input_len - 1)

    seq_data: Dict[int, SequenceData] = {0: seq}

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
        request_id=random_uuid(),
        is_prompt=is_prompt,
        seq_data=seq_data,
        sampling_params=sampling_params,
        block_tables=block_tables,
        do_sample=do_sample,
        # Assume the rest doesn't matter and uses defaults
    )
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
    vllm_args = ("--model meta-llama/Llama-3.1-8B-Instruct "
                 f"-tp {1} "
                 f"-pp {2} "
                 "--collect-detailed-traces worker").split()
    vllm_args = parser.parse_args(vllm_args)

    benchmark_batch_param = BenchmarkBatchParam(
        prefill_input_lens=[1024, 1024],
        decode_input_lens=[128 for _ in range(512)],
        log_dir='./logs',
        gpu_freq_mhz=nvml_get_available_freq()[0],
        delay_time_s=0.0,
    )

    uvloop.run(benchmark_batch(vllm_args, [benchmark_batch_param]))
