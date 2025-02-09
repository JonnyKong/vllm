# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from abc import ABC, abstractmethod

from vllm.config import VllmConfig
from vllm.platforms.nvml_utils import nvml_set_freq


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

    def step(self) -> None:
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

    @abstractmethod
    def adjust(self) -> int:
        pass

    async def set_freq_async(self, frequency: int) -> None:
        await asyncio.to_thread(nvml_set_freq, frequency)

    @staticmethod
    def create_from_config(config: VllmConfig,
                           llm_engine) -> 'NvmlFreqModulator':
        '''
        Factory method to create an NvmlFreqModulator instance from a
        VllmConfig. Currently, always returns a RuleBasedNvmlFreqModulator.
        '''
        return RuleBasedNvmlFreqModulator(llm_engine, interval_s=1.0)


class RuleBasedNvmlFreqModulator(NvmlFreqModulator):
    '''
    A rule-based implementation of NvmlFreqModulator. Adjusts the GPU frequency
    based on the fraction of active tasks in the scheduler.
    '''

    def __init__(self, llm_engine, interval_s: float) -> None:
        super().__init__(llm_engine, interval_s)

    frequency_table = {
        (0.0, 0.2): 825,
        (0.2, 0.5): 1125,
        (0.5, 0.8): 1440,
        (0.8, 1.0): 1740,
    }

    def adjust(self) -> int:
        running_tasks = len(self.llm_engine.scheduler[0].running)
        max_tasks = self.llm_engine.scheduler[0].scheduler_config.max_num_seqs
        fraction = running_tasks / max_tasks if max_tasks > 0 else 0

        for (low, high), freq in self.frequency_table.items():
            if low <= fraction < high:
                return freq

        return max(self.frequency_table.values())
