# SPDX-License-Identifier: Apache-2.0
import math
import multiprocessing
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vllm.logger import init_logger
from vllm.platforms.nvml_utils import timeit

from .q_learning_nvml_freq_modulator import QLearningNvmlFreqModulator

logger = init_logger(__name__)


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNNvmlFreqModulator(QLearningNvmlFreqModulator):

    def __init__(self,
                 llm_engine,
                 interval_s: float,
                 freq_choices: List[int],
                 power_usage_queue: multiprocessing.SimpleQueue,
                 log_dir: str,
                 save_rl_history: bool = False,
                 pretrained_rl_model_path: Optional[str] = None,
                 gpu_tdp: int = 300,
                 alpha: float = 0.001,
                 gamma: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 1000,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 10,
                 tbt_slo: float = 0.5):
        super().__init__(llm_engine, interval_s, freq_choices,
                         power_usage_queue, log_dir, save_rl_history,
                         pretrained_rl_model_path, gpu_tdp)
        self.alpha = alpha  # Learning rate

        self.gamma = gamma  # Discount factor
        self.epsilon_start = epsilon_start  # Initial exploration rate
        self.epsilon_end = epsilon_end  # Final exploration rate
        self.epsilon_decay_steps = epsilon_decay_steps  # Steps to decay epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory: deque = deque(maxlen=memory_size)
        self.tbt_slo = tbt_slo
        self.action_size: int = len(freq_choices)
        self.state_size: int = 1
        self.device: torch.device = torch.device("cpu")

        self.policy_net = DQN(self.state_size,
                              self.action_size).to(self.device)
        self.target_net = DQN(self.state_size,
                              self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.epsilon = self.epsilon_start

        self.previous_state: Optional[np.ndarray] = None
        self.latest_loss: Optional[float] = None
        self._load_model()

    def adjust(self) -> int:
        sys_stats = self.get_sys_stats()
        sys_stats['mean_power_usage'] = self.get_latest_power_reading()

        state = self._get_state(sys_stats)
        action, is_explore = self._select_action(state)
        reward_dict = self._get_reward_dict(sys_stats)
        reward = sum(reward_dict.values())

        if (self.llm_engine.circuit_breaker
                and self.llm_engine.circuit_breaker.is_tripped):
            # If circuit breaker tripped, use highest frequency and do not learn
            self.previous_state = state
            return max(self.freq_choices)

        if self.previous_state is not None:
            self.memory.append((self.previous_state, action, reward, state))

        if len(self.memory) >= self.batch_size:
            with timeit('DQN optimize'):
                self.latest_loss = self._optimize_model()

        if self.step_id % self.target_update == 0:
            with timeit('DQN copy policy net to target net'):
                self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.step_id % 100 == 0:
            with timeit('DQN save model'):
                self._save_model()

        if self.save_rl_history:
            self.rl_history.append({
                'time': time.perf_counter(),
                'step_id': self.step_id,
                'state': state,
                'action': action,
                'is_explore': is_explore,
                'epsilon': self.epsilon,
                'latest_loss': self.latest_loss,
                **reward_dict,
                'reward_total': reward,
            })
            # Save reward history periodically
            if self.step_id % 10 == 0:
                with timeit('DQN save rewards'):
                    self._save_rewards(Path(self.log_dir) / 'rewards.csv')

        self.step_id += 1
        self.previous_state = state
        self._update_epsilon()

        return self.freq_choices[action]

    def _get_state(self, sys_stats: Dict):
        state = np.array([sys_stats['gpu_kv_cache_usage']], dtype=np.float32)
        return state

    def _select_action(self, state: np.ndarray) -> Tuple[int, bool]:
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            is_explore = True
        else:
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax().item()
            is_explore = False
        return action, is_explore

    def _get_reward_dict(self, sys_stats: Dict) -> Dict[str, float]:
        power_reward = 2 - sys_stats['mean_power_usage'] / self.gpu_tdp
        wait_queue_penalty = (-1 if len(self.llm_engine.scheduler[0].waiting)
                              > 5 else 0)
        tbt_penalty = -10 if sys_stats['tbt_mean'] > self.tbt_slo else 0
        return {
            'power_reward': power_reward,
            'wait_queue_penalty': wait_queue_penalty,
            'tbt_penalty': tbt_penalty,
        }

    def _optimize_model(self) -> float:
        """
        Optimize the model and return the loss value.
        """
        batch = random.sample(self.memory, self.batch_size)
        previous_states, actions, rewards, states = zip(*batch)

        previous_states = torch.tensor(previous_states,
                                       device=self.device,
                                       dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device,
                               dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards,
                               device=self.device,
                               dtype=torch.float32).unsqueeze(1)
        states = torch.tensor(states, device=self.device, dtype=torch.float32)

        q_values = self.policy_net(previous_states).gather(1, actions)
        next_q_values = self.target_net(states).max(1, keepdim=True)[0]
        target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _load_model(self):
        if self.pretrained_rl_model_path is not None \
                and Path(self.pretrained_rl_model_path).exists():
            self.policy_net.load_state_dict(
                torch.load(self.pretrained_rl_model_path,
                           map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info('Model loaded from %s', self.pretrained_rl_model_path)
        else:
            logger.info('No pretrained model found at: %s',
                        self.pretrained_rl_model_path)

    def _save_model(self):
        model_path = Path(self.log_dir) / 'dqn_model.pth'
        torch.save(self.policy_net.state_dict(), model_path)
        logger.info('Model saved to: %s', model_path)

    def _update_epsilon(self):
        self.epsilon = self.epsilon_end + (
            (self.epsilon_start - self.epsilon_end) *
            math.exp(-self.step_id / self.epsilon_decay_steps))

    def __del__(self):
        torch.save(self.policy_net.state_dict(),
                   Path(self.log_dir) / 'dqn_model.pth')
