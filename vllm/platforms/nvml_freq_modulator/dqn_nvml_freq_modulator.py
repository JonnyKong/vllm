# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import random
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vllm.logger import init_logger

from .nvml_freq_modulator import NvmlFreqModulator

logger = init_logger(__name__)


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNNvmlFreqModulator(NvmlFreqModulator):

    def __init__(self,
                 llm_engine,
                 interval_s: float,
                 freq_choices: List[int],
                 power_usage_queue: multiprocessing.Queue,
                 log_dir: str,
                 pretrained_rl_model_path: Optional[str] = None,
                 gpu_tdp: int = 300,
                 alpha: float = 0.001,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 10,
                 save_rl_history: bool = False,
                 TBT_SLO: float = 0.5):
        super().__init__(llm_engine, interval_s)
        self.freq_choices = freq_choices
        self.power_usage_queue = power_usage_queue
        self.log_dir = log_dir
        self.pretrained_rl_model_path = pretrained_rl_model_path
        self.gpu_tdp = gpu_tdp
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory: deque = deque(maxlen=memory_size)
        self.save_rl_history = save_rl_history
        self.TBT_SLO = TBT_SLO
        self.step_id: int = 0
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

        self._load_model()

    def adjust(self) -> int:
        start_time = time.time()
        sys_stats = self.get_sys_stats()
        state = np.array([sys_stats['gpu_kv_cache_usage']], dtype=np.float32)
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)

        action = self._select_action(state_tensor)
        mean_power_usage = 0
        while not self.power_usage_queue.empty():
            mean_power_usage = self.power_usage_queue.get()

        power_reward = 2 - mean_power_usage / self.gpu_tdp
        wait_queue_penalty = -1 if len(
            self.llm_engine.scheduler[0].waiting) > 5 else 0
        TBT_penalty = -10 if sys_stats['tbt_mean'] > self.TBT_SLO else 0
        reward = power_reward + wait_queue_penalty + TBT_penalty

        next_state = np.array([sys_stats['gpu_kv_cache_usage']],
                              dtype=np.float32)
        self.memory.append((state, action, reward, next_state))

        self._optimize_model()

        if self.step_id % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.step_id % 100 == 0:
            self._save_model()

        self.step_id += 1
        end_time = time.time()
        print(f"Step {self.step_id} took {end_time - start_time} seconds")
        return self.freq_choices[action]

    def _select_action(self, state_tensor):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            return self.policy_net(state_tensor).argmax().item()

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device,
                               dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards,
                               device=self.device,
                               dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states,
                                   device=self.device,
                                   dtype=torch.float32)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _load_model(self):
        if self.pretrained_rl_model_path is not None \
                and Path(self.pretrained_rl_model_path).exists():
            self.policy_net.load_state_dict(
                torch.load(self.pretrained_rl_model_path,
                           map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Model loaded from", self.pretrained_rl_model_path)
        else:
            print("No pretrained model found at ",
                  self.pretrained_rl_model_path)

    def _save_model(self):
        model_path = Path(self.log_dir) / 'dqn_model.pth'
        torch.save(self.policy_net.state_dict(), model_path)
        print("Model saved to", model_path)

    def __del__(self):
        torch.save(self.policy_net.state_dict(),
                   Path(self.log_dir) / 'dqn_model.pth')
