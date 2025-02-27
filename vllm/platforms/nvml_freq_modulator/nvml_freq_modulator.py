# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from vllm.engine.metrics_types import Stats
from vllm.logger import init_logger
from vllm.platforms.nvml_utils import nvml_set_freq

logger = init_logger(__name__)


class NvmlFreqModulator(ABC):
    '''
    Base class for a GPU frequency modulator using NVML. Adjusts the GPU
    frequency at specified intervals by invoking the `adjust` method, which
    must be implemented by subclasses.

    TODO: adjust each GPU separately.
    '''

    def __init__(self, llm_engine, interval_s: float) -> None:
        self.llm_engine = llm_engine
        self.interval_s = interval_s
        self.last_adjustment_time = time.perf_counter()

        # stats over the course of a RL step
        self.stats_buffer: List[Stats] = []

    def step(self, stats: Optional[Stats]) -> None:
        """
        Should be called on each engine step, passing in the stats at that
        step.
        """
        if stats:
            self.stats_buffer.append(stats)

        current_time = time.perf_counter()
        if current_time - self.last_adjustment_time >= self.interval_s:
            freq = self.adjust()
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(asyncio.create_task,
                                          self.set_freq_async(freq))
            except RuntimeError:
                # Entering here means we are in a non-async context, and not
                # nested inside another event loop
                asyncio.run(self.set_freq_async(freq))
            self.last_adjustment_time = current_time
            self.stats_buffer = []

    @abstractmethod
    def adjust(self) -> int:
        pass

    async def set_freq_async(self, frequency: int) -> None:
        await asyncio.to_thread(nvml_set_freq, frequency)

    def get_sys_stats(self) -> Dict[str, float]:
        """
        Extract states potentially usable by RL.
        """
        # TBT
        tbt_arr = []
        for s in self.stats_buffer:
            tbt_arr.extend(s.time_per_output_tokens_iter)

        # Num tokens decoded
        num_tokens_decoded = 0
        for s in self.stats_buffer:
            num_tokens_decoded += len(s.time_to_first_tokens_iter)
            num_tokens_decoded += len(s.time_per_output_tokens_iter)

        return {
            'running_req_cnt':
            sum(len(s.running) for s in self.llm_engine.scheduler),
            'running_req_max':
            sum(s.scheduler_config.max_num_seqs
                for s in self.llm_engine.scheduler),
            'waiting_req_cnt':
            sum(len(s.waiting) for s in self.llm_engine.scheduler),
            'waiting_token_cnt':
            sum(r.first_seq.get_prompt_len()
                for scheduler in self.llm_engine.scheduler
                for r in scheduler.waiting),
            'gpu_kv_cache_usage':
            self._get_gpu_kv_cache_usage(),
            'tbt_mean':
            float(np.mean(tbt_arr)) if len(tbt_arr) > 0 else 0.0,
            'tbt_p99':
            float(np.percentile(tbt_arr, 99)) if len(tbt_arr) > 0 else 0.0,
            'num_tokens_decoded':
            num_tokens_decoded,
        }

    def _get_gpu_kv_cache_usage(self):
        # https://github.com/vllm-project/vllm/blob/1f0ae3ed0aa9af7f5e88e56f5c960cc919c2f090/vllm/engine/llm_engine.py#L1581-L1588
        num_total_gpu = self.llm_engine.cache_config.num_gpu_blocks
        gpu_cache_usage_sys = 0.
        if num_total_gpu:  # Guard against both None and 0
            num_free_gpu = sum(
                scheduler.block_manager.get_num_free_gpu_blocks()
                for scheduler in self.llm_engine.scheduler)
            gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)
        return gpu_cache_usage_sys
