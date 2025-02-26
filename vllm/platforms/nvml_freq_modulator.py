# SPDX-License-Identifier: Apache-2.0
import asyncio
import math
import multiprocessing
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from vllm.config import VllmConfig
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
        elif config.freq_mod_mode == 'q-learn':
            assert config.log_dir
            a40_freq_choices = [
                540, 660, 780, 900, 1020, 1140, 1260, 1380, 1500, 1620, 1740
            ]
            log_file = Path(config.log_dir) / 'q_learning.csv'
            return QLearningNvmlFreqModulator(
                llm_engine,
                interval_s=interval_s,
                freq_choices=a40_freq_choices,
                log_file=str(log_file),
                power_usage_queue=llm_engine.power_usage_queue)
        else:
            raise NotImplementedError(
                f'Unrecognized freq_mod_mode: {llm_engine.freq_mod_mode}')

    def get_sys_stats(self) -> Dict[str, float]:
        """
        Extract states potentially usable by RL.
        """
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
        (0.5, 0.9): 1440,
        (0.9, 1.0): 1740,
    }

    def adjust(self) -> int:
        sys_stats = self.get_sys_stats()

        fraction = sys_stats['gpu_kv_cache_usage']

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


class QLearningNvmlFreqModulator(NvmlFreqModulator):
    """
    A GPU frequency modulator using tabular Q-learning to adjust the GPU
    frequency based on the state of the system.
    """

    def __init__(self,
                 llm_engine,
                 interval_s: float,
                 freq_choices: List[int],
                 power_usage_queue: multiprocessing.SimpleQueue,
                 log_file: str,
                 gpu_tdp: int = 300,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1) -> None:
        super().__init__(llm_engine, interval_s)
        self.freq_choices = freq_choices
        self.action_list = [i for i in range(len(freq_choices))]
        self.current_freq = max(freq_choices)
        self.log_file = log_file
        self.q_table: Dict[float, Dict[int, float]] = {}
        self.load_q_table()

        self.gpu_tdp: float = gpu_tdp
        self.alpha: float = alpha  # Learning rate
        self.gamma: float = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.previous_state: Optional[float] = None
        self.previous_action: Optional[int] = None
        self.power_usage_queue = power_usage_queue

    def adjust(self) -> int:
        sys_stats = self.get_sys_stats()
        kv_usage = sys_stats['gpu_kv_cache_usage']  # already 0 to 1.0

        # 1.0 means (0.9 to 1.0]
        # 0.1 means (0.0 to 0.1]
        # 0.0  means [0.0]
        state = math.ceil(kv_usage * 10) / 10

        mean_power_usage = 0
        while not self.power_usage_queue.empty():
            # takes the latest value only
            mean_power_usage = self.power_usage_queue.get()

        power_reward = 1 - mean_power_usage / self.gpu_tdp
        # TODO, check if penalty should be higher or lower
        wait_queue_penalty = -1 if len(
            self.llm_engine.scheduler[0].waiting) > 0 else 0
        print('============= wait_queue_penalty: ', wait_queue_penalty)

        if self.previous_state is not None and self.previous_action is not None:
            reward = power_reward + wait_queue_penalty  #reward function
            self._update_q_table(self.previous_state, self.previous_action,
                                 reward, state)

        action = self._select_action(state)
        self.previous_state = state
        self.previous_action = action

        asyncio.create_task(asyncio.to_thread(self.save_data))

        return self.freq_choices[action]

    def _select_action(self, state: float) -> int:
        if random.uniform(0, 1) < self.epsilon and state != 1.0:
            return random.choice(self.action_list)  # Explore
        return self._get_best_action(state)  # Exploit

    def _get_best_action(self, state: float) -> int:
        state_actions = self.q_table.get(state, {})
        if not state_actions:
            return random.choice(self.action_list)
        return max(state_actions, key=lambda x: state_actions[x])

    def _update_q_table(self, state: float, action: int, reward: float,
                        next_state: float) -> None:
        state_actions = self.q_table.setdefault(
            state, {freq: 0
                    for freq in self.action_list})
        next_state_actions = self.q_table.get(
            next_state, {freq: 0
                         for freq in self.action_list})
        best_next_action = max(next_state_actions,
                               key=lambda x: float(next_state_actions[x]))
        td_target = reward + self.gamma * next_state_actions[best_next_action]
        td_error = td_target - state_actions[action]
        state_actions[action] += self.alpha * td_error

    def initialize_q_table(self, default_value: float) -> None:
        for state in [i / 10 for i in range(11)
                      ]:  # states are 0.0, 0.1, 0.2, ..., 1.0
            self.q_table[state] = {
                freq: default_value
                for freq in self.action_list
            }
        self.q_table[1.0] = {
            freq: 0
            for freq in [self.action_list[10]]
        }  # only the highest frequency is allowed when the queue is full
        suggested_q_value = 1
        self.q_table[0.0][self.action_list[0]] = suggested_q_value
        self.q_table[0.1][self.action_list[1]] = suggested_q_value
        self.q_table[0.2][self.action_list[2]] = suggested_q_value
        self.q_table[0.3][self.action_list[3]] = suggested_q_value
        self.q_table[0.4][self.action_list[4]] = suggested_q_value
        self.q_table[0.5][self.action_list[5]] = suggested_q_value
        self.q_table[0.6][self.action_list[6]] = suggested_q_value
        self.q_table[0.7][self.action_list[7]] = suggested_q_value
        self.q_table[0.8][self.action_list[8]] = suggested_q_value
        self.q_table[0.9][self.action_list[9]] = suggested_q_value

    def load_q_table(self) -> None:
        try:
            df = pd.read_csv(self.log_file + '_init')
            for _, row in df.iterrows():
                state = row['state']
                action = row['action']
                q_value = row['q_value']
                if state not in self.q_table:
                    self.q_table[state] = {}
                self.q_table[state][action] = q_value
            logger.info('Q-table loaded from %s', self.log_file)
        except FileNotFoundError:
            logger.info('File %s not found. Using default values.',
                        self.log_file)
            self.initialize_q_table(0)

    def save_data(self) -> None:
        df = pd.DataFrame([(state, action, q_value)
                           for state, actions in self.q_table.items()
                           for action, q_value in actions.items()],
                          columns=['state', 'action', 'q_value'])
        df.to_csv(self.log_file, index=False)

    def __del__(self):
        self.save_data()
