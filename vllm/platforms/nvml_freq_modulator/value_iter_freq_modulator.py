# SPDX-License-Identifier: Apache-2.0
import random
import time
from typing import List, Optional, Tuple

from vllm.logger import init_logger

from .nvml_freq_modulator import NvmlFreqModulator

logger = init_logger(__name__)


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
        self.iteration = 0  # Counter for periodic logging

        self.log_file_handle = open(log_file, 'w')  # noqa
        self.log_file_handle.write("timestamp,state,action,next_state\n")

    def adjust(self) -> int:
        sys_stats = self.get_sys_stats()
        logger.debug('sys_stats: %s', str(sys_stats))

        # State is sum of request count in running and waiting queue,
        # normalized by running queue size, which can go beyond 1.0
        state = (sys_stats['running_req_cnt'] +
                 sys_stats['waiting_req_cnt']) / sys_stats['running_req_max']

        timestamp = time.perf_counter()

        if self.previous_state is not None:
            if (self.llm_engine.circuit_breaker
                    and self.llm_engine.circuit_breaker.is_tripped):
                # The state transition model assumes a specific request arrival
                # rate. When the circuit breaker is tripped, the request rate
                # is 0, so stop logging data
                pass
            else:
                self.data_log.append(
                    (timestamp, self.previous_state, self.current_freq, state))

        self.current_freq = self._select_action(state)
        self.previous_state = state

        # Periodically save every N iterations
        if self.iteration % 5 == 0 and self.data_log:
            self._save_incremental()

        return self.current_freq

    def _select_action(self, state: float) -> int:
        return random.choice(self.frequency_list)

    def _save_incremental(self) -> None:
        self.log_file_handle.writelines(
            f"{ts},{prev_state},{action},{next_state}\n"
            for ts, prev_state, action, next_state in self.data_log)
        self.log_file_handle.flush()
        self.data_log.clear()

    def __del__(self):
        """ Ensure the file handle is closed properly on object destruction. """
        self.log_file_handle.close()
