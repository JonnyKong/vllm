# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

from vllm.logger import init_logger

logger = init_logger(__name__)


class CircuitBreaker(ABC):

    def __init__(self, llm_engine) -> None:
        self.llm_engine = llm_engine
        self._is_tripped: bool = False

    @abstractmethod
    def step(self):
        pass

    @property
    def is_tripped(self):
        return self._is_tripped


class SimpleCircuitBreaker(CircuitBreaker):

    def __init__(self,
                 llm_engine,
                 mode: str,
                 thresh_low: float = 0.9,
                 thresh_high: float = 0.98,
                 trigger_count_threshold: int = 100):
        super().__init__(llm_engine)
        assert mode in ['running_queue_util', 'gpu_kv_cache_util']

        self.mode = mode
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        self.trigger_count_threshold = trigger_count_threshold

        self.trigger_count = 0

    def step(self):
        if self.mode == 'running_queue_util':
            util = self.get_running_queue_util()
        elif self.mode == 'gpu_kv_cache_util':
            util = self.get_gpu_kv_cache_util()
        else:
            raise NotImplementedError()

        if not self._is_tripped and util > self.thresh_high:
            self.trigger_count += 1
            if self.trigger_count >= self.trigger_count_threshold:
                self._is_tripped = True
                logger.info("Circuit breaker tripped")
        elif util < self.thresh_low:
            self.trigger_count = 0
            if self._is_tripped:
                self._is_tripped = False
                logger.info("Circuit breaker reset")

    def get_running_queue_util(self):
        """
        The utilization is calculated as the ratio of the total number of
        running and waiting sequences, to the maximum number of sequences
        allowed across all schedulers, and therefore can go beyond 100%.
        """
        num_running = sum(
            len(scheduler.running) for scheduler in self.llm_engine.scheduler)
        num_waiting = sum(
            len(scheduler.waiting) for scheduler in self.llm_engine.scheduler)
        num_running_max = sum(scheduler.scheduler_config.max_num_seqs
                              for scheduler in self.llm_engine.scheduler)
        running_queue_util = (num_running + num_waiting) / num_running_max
        return running_queue_util

    def get_gpu_kv_cache_util(self):
        num_total_gpu = self.llm_engine.cache_config.num_gpu_blocks
        gpu_cache_usage_sys = 0.
        if num_total_gpu:  # Guard against both None and 0
            num_free_gpu = sum(
                scheduler.block_manager.get_num_free_gpu_blocks()
                for scheduler in self.llm_engine.scheduler)
            gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)
            return gpu_cache_usage_sys
        else:
            return 0.0
