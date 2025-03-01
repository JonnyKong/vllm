# SPDX-License-Identifier: Apache-2.0
import asyncio
import multiprocessing
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from vllm.logger import init_logger

from .nvml_freq_modulator import NvmlFreqModulator

logger = init_logger(__name__)


class QLearningNvmlFreqModulator(NvmlFreqModulator):
    """
    A GPU frequency modulator using tabular Q-learning to adjust the GPU
    frequency based on the state of the system.
    """

    def __init__(
        self,
        llm_engine,
        interval_s: float,
        freq_choices: List[int],
        power_usage_queue: multiprocessing.SimpleQueue,
        log_dir: str,
        save_rl_history: bool = False,
        pretrained_rl_model_path: Optional[str] = None,
        gpu_tdp: int = 300,
    ):
        super().__init__(llm_engine, interval_s)

        self.freq_choices = freq_choices
        self.power_usage_queue = power_usage_queue
        self.log_dir = log_dir
        self.save_rl_history = save_rl_history
        self.pretrained_rl_model_path = pretrained_rl_model_path
        self.gpu_tdp = gpu_tdp

        # [time, step, state, action, reward] at each step
        self.rl_history: List[List] = []

        # RL step number
        self.step_id = 0

        (Path(self.log_dir) / 'rewards.csv').unlink(missing_ok=True)

    @abstractmethod
    def adjust(self) -> int:
        pass

    def get_latest_power_reading(self):
        mean_power_usage = 0
        while not self.power_usage_queue.empty():
            # takes the latest value only
            mean_power_usage = self.power_usage_queue.get()
        return mean_power_usage

    def _save_rewards(self, log_file: Union[str, Path]):
        """Non-blocking method to trigger async saving."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_rewards_async(log_file))
        except RuntimeError:
            # No event loop running; run the async function synchronously
            asyncio.run(self._save_rewards_async(log_file))

    async def _save_rewards_async(self, log_file: Union[str, Path]):
        """Asynchronously appends new rewards to the CSV file."""
        log_file = Path(log_file)
        append_mode = log_file.exists()

        df = pd.DataFrame(
            self.rl_history,
            columns=['time', 'step', 'state', 'action', 'reward'])

        await asyncio.to_thread(
            df.to_csv,
            log_file,
            mode='a',
            index=False,
            # Write header only if the file does not exist
            header=not append_mode,
        )

        # Clear memory after writing
        self.rl_history.clear()
