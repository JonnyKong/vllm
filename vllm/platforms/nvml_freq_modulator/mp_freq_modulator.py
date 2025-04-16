# SPDX-License-Identifier: Apache-2.0
import copy
from itertools import product
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import msgspec
import numpy as np
from lightgbm import Booster

from vllm.engine.metrics_types import Stats
from vllm.logger import init_logger

from .nvml_freq_modulator import NvmlFreqModulatorInterface

logger = init_logger(__name__)


class FreqModMsg(msgspec.Struct):
    """
    Msg from client to server.
    """
    num_prefills: int


class MPNvmlFreqModulatorClient(NvmlFreqModulatorInterface):
    """
    Adjusts frequency in a separate process. Useful if the procedure of
    determining the frequency is computation heavy.
    """

    def __init__(
        self,
        llm_engine,
        freq_choices: list[int],
    ):
        self.llm_engine = llm_engine

        self.encoder = msgspec.msgpack.Encoder()

        self.q: Queue = Queue()
        self.server = _MPNvmlFreqModulatorServer(freq_choices, self.q,
                                                 "./assets")
        self.server_process = Process(target=self.server.run, daemon=True)
        self.server_process.start()

    def step(self, stats: Optional[Stats]) -> None:
        msg = self.build_msg()
        msg_encoded = self.encoder.encode(msg)
        self.q.put(msg_encoded)

    @staticmethod
    def build_msg() -> FreqModMsg:
        return FreqModMsg(num_prefills=0, )


class _MPNvmlFreqModulatorServer:

    def __init__(
        self,
        freq_choices: list[int],
        q: Queue,
        path_to_models: str,
        future_window: int = 2,
        tbt_sla: float = 0.25,
        ttft_sla: float = 1.0,
    ):
        self.freq_choices = freq_choices
        self.q = q

        self.future_windows = future_window
        self.tbt_sla = tbt_sla
        self.ttft_sla = ttft_sla

        latency_model_dir = Path(path_to_models)
        power_model_dir = Path(path_to_models)

        self.power_model = Booster(model_file=power_model_dir /
                                   'power_model.txt')
        self.latency_model_prefill = Booster(model_file=latency_model_dir /
                                             'latency_model_prefill-only.txt')
        self.latency_model_decode = Booster(model_file=latency_model_dir /
                                            'latency_model_decode-only.txt')
        self.latency_model_hybrid = Booster(model_file=latency_model_dir /
                                            'latency_model_hybrid.txt')

        self.decoder = msgspec.msgpack.Decoder()

    def run(self):
        while True:
            states: FreqModMsg = self.decoder.decode(self.q.get())

            future_states, prefill_cycles = self.get_future_states(
                states, self.future_windows)

            num_waiting_reqs = len(
                states['wait_queue_num_prefill_tokens_per_req'])
            # Smaller if not all requests are prefilled in `future_windows`
            assert len(prefill_cycles) <= num_waiting_reqs

            # selected_freq = self._get_next_freq_brute_force(env, future_obs, info, prefill_cycles)
            selected_freq = self._get_next_freq_dp(future_states,
                                                   prefill_cycles)
            print(f"Selected freq: {selected_freq}")

    def _get_next_freq_dp(self, future_states, prefill_cycles):
        """
        - Initialization:
            - use the highest freq for every future window
        - Loop through i=1 to N-1:
            - Assuming we have the optimal solution using ONLY the first i
              avail freqs, we derive the optimal solution using the first i+1
              avail freqs by:
                  - Try flipping each freq selection that is currently
                    freqs_avail[i] to freqs_avail[i+1] . This may give up to
                    2^K candidate solutions. Keep the solution if it meets TTFT
                    & TBT SLA
                  - Select among the candidate solutions the optimal, i.e.
                    having the lowest energy
        """
        freq_choices_desc = sorted(copy.deepcopy(self.freq_choices),
                                   reverse=True)

        max_future_vision = self.future_windows

        # Pre-compute latency and power for each future window for each freq
        lat_mat_list = []
        power_mat_list = []
        for window_idx in range(max_future_vision):
            lat_mat_list.append(
                self.predict_latencies(future_states[window_idx],
                                       freq_choices_desc))
        power_mat_list = self.predict_powers_future_states(
            future_states, freq_choices_desc)
        lat_mat = np.array(lat_mat_list)
        power_mat = np.array(power_mat_list)
        energy_mat = lat_mat * power_mat
        assert lat_mat.shape == (max_future_vision, len(freq_choices_desc))
        assert power_mat.shape == (max_future_vision, len(freq_choices_desc))

        # Start with the highest freq for each window
        selected_freq_ids = [0 for _ in range(self.future_windows)]

        for freq_idx in range(1, len(freq_choices_desc)):
            # Collect the candidates from `selected_freqs`
            candidates_: list[list[int]] = [[]]
            for window_idx in range(max_future_vision):
                if selected_freq_ids[window_idx] == freq_idx - 1:
                    freq_ids_this_window = [freq_idx - 1, freq_idx]
                else:
                    freq_ids_this_window = [selected_freq_ids[window_idx]]
                candidates_ = [[
                    *c, f
                ] for c, f in product(candidates_, freq_ids_this_window)]
            candidates = np.array(candidates_)
            # [n_candidates, max_future_vision]
            assert candidates.shape[1] == max_future_vision

            # Keep candidates that meet TBT SLA
            # Compute TBT for all candidates in parallel
            tbt_arr = lat_mat[np.arange(max_future_vision)[:, None],
                              candidates.T]
            max_tbt_per_candidate = np.max(tbt_arr, axis=0)
            sla_tbt_mask = max_tbt_per_candidate <= self.tbt_sla

            # Compute TTFT for all candidates in parallel
            time_till_finish_per_batch = np.cumsum(tbt_arr, axis=0)
            time_till_finish_per_req = time_till_finish_per_batch[
                np.array(prefill_cycles, dtype=int) - 1, :]
            waiting_time_per_req = np.array(
                future_states[0]['wait_queue_waiting_time_per_req']
                [:len(prefill_cycles)])[:, None]
            ttft_arr = time_till_finish_per_req + waiting_time_per_req
            sla_ttft_mask = np.all(ttft_arr <= self.ttft_sla, axis=0)

            # Combine masks to filter valid candidates
            valid_mask = sla_tbt_mask & sla_ttft_mask
            candidates = candidates[valid_mask]

            # Select the min-energy candidate as `selected_freq_ids`
            if len(candidates) > 0:
                candidates = np.array(candidates)
                energy_per_batch = energy_mat[
                    np.arange(max_future_vision)[:, None], candidates.T]
                total_energy = np.sum(energy_per_batch, axis=0)
                selected_freq_ids = candidates[np.argmin(total_energy)]
            else:
                break

        selected_freq = freq_choices_desc[selected_freq_ids[0]]
        return self.freq_choices.index(selected_freq)

    def get_future_states(self, states,
                          future_window: int) -> tuple[list, list]:
        """
        Get the future observation for the given index.
        The future observation is a list of observations for the next `future_windows` batches.
        Assumptions:
            - prefills are (poorly) chunked now
            - rough waiting request using a sampled distribution
            - no requests reach EOS during future calculations
            - prefill budget hard coded to 1024 (for now)
        """
        # A list that tells you for each request in the wait queue
        # how many iterations it will take to get the first token
        prefill_cycles = []

        num_prefills = states['num_prefills']
        num_decodes = states['num_decodes']
        prefill_len_sum = states['prefill_len_sum']
        prefill_len_max = states['prefill_len_max']
        prefill_len_std = states['prefill_len_std']
        decode_len_sum = states['decode_len_sum']
        decode_len_max = states['decode_len_max']
        decode_len_std = states['decode_len_std']

        # Construct a dummy wait queue to simulate future chunked prefills
        num_prefill_tokens = states['wait_queue_num_prefill_tokens_per_req']
        num_processed_tokens = states[
            'wait_queue_num_processed_tokens_per_req']
        waiting_times = states['wait_queue_waiting_time_per_req']

        dummy_wait_queue = [
            m - n for m, n in zip(num_prefill_tokens, num_processed_tokens)
        ]

        future_states = [copy.deepcopy(states) for _ in range(future_window)]
        for i in range(1, future_window):
            decode_len_max = max(decode_len_max, prefill_len_max)
            decode_len_std = np.sqrt((num_decodes * decode_len_std**2 +
                                      num_prefills * prefill_len_std**2) /
                                     (num_decodes + num_prefills + 1e-6))
            # all requests progress by 1 and prefills are added to decodes
            decode_len_sum += num_decodes + prefill_len_sum
            # last batches prefills are added to decodes
            num_decodes += num_prefills

            # Chunked prefill logic
            budget_left = 1024
            prefills = []
            while budget_left > 0 and len(dummy_wait_queue) > 0:
                num_tokens = min(budget_left, dummy_wait_queue[0])
                prefills.append(num_tokens)

                budget_left -= num_tokens
                dummy_wait_queue[0] -= num_tokens
                if dummy_wait_queue[0] == 0:
                    dummy_wait_queue.pop(0)
                    prefill_cycles.append(i + 1)

            num_prefills = len(prefills)
            if num_prefills > 0:
                prefill_len_sum = np.sum(prefills)
                prefill_len_max = np.max(prefills, initial=prefills[0])
                prefill_len_std = np.std(prefills)
            else:
                prefill_len_sum = 0
                prefill_len_max = 0
                prefill_len_std = 0

            future_states[i]['num_decodes'] = num_decodes
            future_states[i]['decode_len_sum'] = decode_len_sum
            future_states[i]['decode_len_max'] = decode_len_max
            future_states[i]['decode_len_std'] = decode_len_std
            future_states[i]['num_prefills'] = num_prefills
            future_states[i]['prefill_len_sum'] = prefill_len_sum
            future_states[i]['prefill_len_max'] = prefill_len_max
            future_states[i]['prefill_len_std'] = prefill_len_std

        return future_states, prefill_cycles

    def predict_latencies(self, states, freq_choices) -> list[float]:
        """
        Predict latency of the upcoming batch for each freq in `freq_choices`.
        """

        num_prefills = states['num_prefills']
        prefill_len_sum = states['prefill_len_sum']
        prefill_len_std = states['prefill_len_std']
        prefill_len_max = states['prefill_len_max']
        num_decodes = states['num_decodes']
        decode_len_sum = states['decode_len_sum']
        decode_len_std = states['decode_len_std']
        decode_len_max = states['decode_len_max']
        running_queue_len = states['running_queue_len']

        # Predict for each freq
        if num_prefills > 0 and num_decodes > 0:
            latency_model = self.latency_model_hybrid
            input = np.array([
                num_prefills, prefill_len_sum, prefill_len_std,
                prefill_len_max, num_decodes, decode_len_sum, decode_len_std,
                decode_len_max
            ])
        elif num_prefills > 0:
            latency_model = self.latency_model_prefill
            input = np.array([
                num_prefills, prefill_len_sum, prefill_len_std, prefill_len_max
            ])
        elif num_decodes > 0:
            latency_model = self.latency_model_decode
            input = np.array(
                [num_decodes, decode_len_sum, decode_len_std, decode_len_max])
        else:
            return [0.0 for _ in range(len(freq_choices))]

        # Predict latency for each freq
        input = np.hstack([
            np.array(freq_choices).reshape(-1, 1),
            np.tile(input, (len(freq_choices), 1)),
        ])
        latency_arr = latency_model.predict(input)
        latency_arr = np.exp(latency_arr)

        cpu_overhead_s = self.get_cpu_overhead_us(running_queue_len) / 1e6
        latency_arr += cpu_overhead_s
        return latency_arr.tolist()

    def predict_powers_future_states(self, future_states,
                                     freq_choices) -> list[list[float]]:
        """
        Predict power of the all batches for each freq in `freq_choices`.
        """
        inputs = []
        for states in future_states:
            num_prefills = states['num_prefills']
            prefill_len_sum = states['prefill_len_sum']
            prefill_len_std = states['prefill_len_std']
            prefill_len_max = states['prefill_len_max']
            num_decodes = states['num_decodes']
            decode_len_sum = states['decode_len_sum']
            decode_len_std = states['decode_len_std']
            decode_len_max = states['decode_len_max']
            input = np.array([
                num_prefills, prefill_len_sum, prefill_len_std,
                prefill_len_max, num_decodes, decode_len_sum, decode_len_std,
                decode_len_max
            ])
            input = np.hstack([
                np.array(freq_choices).reshape(-1, 1),
                np.tile(input, (len(freq_choices), 1)),
            ])
            inputs.append(input)

        # Combine all inputs into a single batch for prediction
        batch_input = np.vstack(inputs)
        power_arr = self.power_model.predict(batch_input)

        # Reshape the output to match the future_states structure
        output_arr = power_arr.reshape(len(future_states), len(freq_choices))
        return output_arr.tolist()

    def get_cpu_overhead_us(self, running_queue_len: int) -> float:
        return np.nan
