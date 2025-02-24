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
    THRESH_LOW = 0.05
    THRESH_HIGH = 1.15

    def step(self):
        num_running = sum(
            len(scheduler.running) for scheduler in self.llm_engine.scheduler)
        num_waiting = sum(
            len(scheduler.waiting) for scheduler in self.llm_engine.scheduler)
        num_running_max = sum(scheduler.scheduler_config.max_num_seqs
                              for scheduler in self.llm_engine.scheduler)
        running_util = (num_running + num_waiting) / num_running_max

        if not self._is_tripped and running_util > self.THRESH_HIGH:
            self._is_tripped = True
            logger.info("Circuit breaker tripped")
        elif self._is_tripped and running_util < self.THRESH_LOW:
            self._is_tripped = False
            logger.info("Circuit breaker reset")
