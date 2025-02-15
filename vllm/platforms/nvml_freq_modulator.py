# SPDX-License-Identifier: Apache-2.0
import asyncio
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

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
        interval_s = 1.0
        if config.freq_mod_mode == 'rule':
            return RuleBasedNvmlFreqModulator(llm_engine,
                                              interval_s=interval_s)
        elif config.freq_mod_mode == 'value-iter':
            assert config.log_dir
            a40_freq_choices = [210, 510, 825, 1125, 1440, 1740]
            log_file = Path(config.log_dir) / 'value_iter.csv'
            return ValueIterationNvmlFreqModulator(
                llm_engine,
                interval_s=interval_s,
                freq_choices=a40_freq_choices,
                log_file=str(log_file))
        else:
            raise NotImplementedError(
                f'Unrecognized freq_mod_mode: {llm_engine.freq_mod_mode}')


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


class ValueIterationNvmlFreqModulator(NvmlFreqModulator):
    """
    A GPU frequency modulator designed solely for collecting training data
    for reinforcement learning. It logs <timestamp, state, action, next_state>
    pairs while running vLLM but does not implement or learn an actual policy.
    """

    def __init__(self, llm_engine, interval_s: float, freq_choices: List[int],
                 log_file: str) -> None:
        super().__init__(llm_engine, interval_s)
        self.frequency_list = freq_choices
        self.current_freq = max(freq_choices)
        self.log_file = log_file
        self.data_log: List[Tuple[float, float, int, float]] = []
        self.previous_state: Optional[float] = None

    def adjust(self) -> int:
        running_tasks = len(self.llm_engine.scheduler[0].running)
        max_tasks = self.llm_engine.scheduler[0].scheduler_config.max_num_seqs
        state = running_tasks / max_tasks if max_tasks > 0 else 0
        timestamp = time.perf_counter()

        if self.previous_state:
            self.data_log.append(
                (timestamp, self.previous_state, self.current_freq, state))

        self.current_freq = self._select_action(state)

        self.previous_state = state

        return self.current_freq

    def _select_action(self, state: float) -> int:
        """
        Selects the next GPU frequency action.
        
        If the running queue's utilization exceeds 90%, it selects the highest
        available frequency to prevent the system from entering an overloaded
        state. Otherwise, it selects a frequency randomly from the available
        options.
        """
        if state > 0.9:
            return max(self.frequency_list)
        return random.choice(self.frequency_list)

    def save_data(self) -> None:
        df = pd.DataFrame(
            self.data_log,
            columns=['timestamp', 'state', 'action', 'next_state'])
        df.to_csv(self.log_file, index=False)

    def __del__(self):
        self.save_data()
