# SPDX-License-Identifier: Apache-2.0
import asyncio
import math
import multiprocessing
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from vllm.logger import init_logger

from .q_learning_nvml_freq_modulator import QLearningNvmlFreqModulator

logger = init_logger(__name__)


class QTableFreqModulator(QLearningNvmlFreqModulator):

    def __init__(self,
                 llm_engine,
                 interval_s: float,
                 freq_choices: List[int],
                 power_usage_queue: multiprocessing.SimpleQueue,
                 log_dir: str,
                 save_rl_history: bool = False,
                 pretrained_rl_model_path: Optional[str] = None,
                 gpu_tdp: int = 300,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 tbt_slo: float = 0.5) -> None:
        super().__init__(llm_engine, interval_s, freq_choices,
                         power_usage_queue, log_dir, save_rl_history,
                         pretrained_rl_model_path, gpu_tdp)

        self.alpha: float = alpha  # Learning rate
        self.gamma: float = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.tbt_slo = tbt_slo  # SLO in seconds

        self.action_list = [i for i in range(len(freq_choices))]
        self.current_freq = max(freq_choices)
        self.q_table: Dict[float, Dict[int, float]] = {}
        self.previous_state: Optional[float] = None
        self.previous_action: Optional[int] = None

        self.history_q_table_dir = Path(self.log_dir) / 'q_tables'
        if self.save_rl_history:
            self.history_q_table_dir.mkdir(exist_ok=True, parents=True)

        self.load_q_table()

    def adjust(self) -> int:
        sys_stats = self.get_sys_stats()
        sys_stats['mean_power_usage'] = self.get_latest_power_reading()

        state = self.get_state(sys_stats)
        reward_dict = self.get_reward_dict(sys_stats)
        reward = sum(reward_dict.values())

        if self.previous_state is not None and self.previous_action is not None:
            self._update_q_table(self.previous_state, self.previous_action,
                                 reward, state)

        action = self._select_action(state)
        self.previous_state = state
        self.previous_action = action

        if self.save_rl_history:
            self.rl_history.append({
                'time': time.perf_counter(),
                'step_id': self.step_id,
                'state': state,
                'action': action,
                **reward_dict,
                'reward_total': reward,
            })
            # Save reward history periodically
            if self.step_id % 10 == 0:
                self._save_rewards(Path(self.log_dir) / 'rewards.csv')

        if not self.save_rl_history:
            log_file = Path(self.log_dir) / 'q_learning.csv'
        else:
            log_file = self.history_q_table_dir \
                    / f'q_learning_{self.step_id:06d}.csv'
        asyncio.create_task(asyncio.to_thread(self._save_q_table, log_file))

        self.step_id += 1
        return self.freq_choices[action]

    def get_reward_dict(self, sys_stats: Dict) -> Dict[str, float]:
        # TODO, check if penalty should be higher or lower
        wait_queue_penalty = -1.0 if len(
            self.llm_engine.scheduler[0].waiting) > 5 else 0.0
        power_reward = 2.0 - sys_stats['mean_power_usage'] / self.gpu_tdp
        tbt_penalty = -10.0 if sys_stats['tbt_mean'] > self.tbt_slo else 0.0
        return {
            'power_reward': power_reward,
            'wait_queue_penalty': wait_queue_penalty,
            'tbt_penalty': tbt_penalty,
        }

    def _select_action(self, state: float) -> int:
        if random.uniform(0, 1) < self.epsilon and state != 1.0:
            return random.choice(self.action_list)  # Explore
        return self._get_best_action(state)  # Exploit

    def _get_best_action(self, state: float) -> int:
        state_actions = self.q_table.get(state, {})
        if not state_actions:
            return random.choice(self.action_list)
        return max(state_actions, key=lambda x: state_actions[x])

    def get_state(self, sys_stats: Dict) -> float:
        # 1.0 means (0.9 to 1.0]
        # 0.1 means (0.0 to 0.1]
        # 0.0  means [0.0]
        state = math.ceil(sys_stats['gpu_kv_cache_usage'] * 10) / 10
        return state

    def initialize_q_table(self, default_value: float) -> None:
        # states are 0.0, 0.1, 0.2, ..., 1.0
        for state in [i / 10 for i in range(11)]:
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
        if self.pretrained_rl_model_path is not None \
                and Path(self.pretrained_rl_model_path).exists():
            df = pd.read_csv(self.pretrained_rl_model_path)
            for _, row in df.iterrows():
                state = row['state']
                action = int(round(row['action']))
                q_value = row['q_value']
                if state not in self.q_table:
                    self.q_table[state] = {}
                self.q_table[state][action] = q_value
            logger.info('Q-table loaded from %s',
                        self.pretrained_rl_model_path)
        else:
            logger.info('Initializing Q-table from scratch ...')
            self.initialize_q_table(0)

    def _save_q_table(self, log_file: Union[str, Path]) -> None:
        df = pd.DataFrame([(state, action, q_value)
                           for state, actions in self.q_table.items()
                           for action, q_value in actions.items()],
                          columns=['state', 'action', 'q_value'])
        df.to_csv(log_file, index=False)

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

    def __del__(self):
        log_file = Path(self.log_dir) / 'q_learning.csv'
        self._save_q_table(log_file)
