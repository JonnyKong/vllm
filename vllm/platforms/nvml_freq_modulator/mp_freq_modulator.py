# SPDX-License-Identifier: Apache-2.0
import copy
import time
from dataclasses import dataclass
from itertools import count, product
from multiprocessing import Process, SimpleQueue
from pathlib import Path
from typing import Optional

import msgspec
import numpy as np
from lightgbm import Booster

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.metrics_types import Stats
from vllm.logger import init_logger
from vllm.platforms.nvml_freq_modulator.nvml_freq_modulator import (
    NvmlFreqModulatorInterface)
from vllm.platforms.nvml_utils import (CSVWriter, get_gpu_name,
                                       get_preselected_freq, nvml_set_freq)
from vllm.utils import get_mp_context

logger = init_logger(__name__)

# Change this accordingly
PATH_TO_MODELS = Path.home() / 'EnergyEfficientServing/vidur/artifacts'


class FreqModMsg(msgspec.Struct):
    """
    Msg from client to server.
    """
    now: float
    running_queue_num_tokens_per_req: list[int]
    wait_queue_num_prefill_tokens_per_req: list[int]
    wait_queue_num_processed_tokens_per_req: list[int]
    wait_queue_waiting_time_per_req: list[float]
    gpu_cache_usage_sys: float
    num_precomputed_tokens_per_req_iter: list[int]

    def __post_init__(self):
        assert len(self.wait_queue_num_prefill_tokens_per_req) == len(
            self.wait_queue_num_processed_tokens_per_req)
        assert len(self.wait_queue_num_prefill_tokens_per_req) == len(
            self.wait_queue_waiting_time_per_req)


@dataclass
class FutureState:
    """
    Represents the system state at a particular point in time.
    """
    num_prefills: int
    prefill_len_sum: int
    prefill_len_max: int
    prefill_len_std: float
    num_decodes: int
    decode_len_sum: int
    decode_len_max: int
    decode_len_std: float
    computed_prefill_len_sum: int
    computed_prefill_len_std: float
    computed_prefill_len_max: int


class MPNvmlFreqModulatorClient(NvmlFreqModulatorInterface):
    """
    Adjusts frequency in a separate process. Useful if the procedure of
    determining the frequency is computation heavy.
    """

    def __init__(
            self,
            llm_engine,
            vllm_config: VllmConfig,
            freq_choices: list[int],
            log_dir: Path,
            mod_interval: int = 1,
            tbt_sla: float = 0.25,
            ttft_sla: float = 1.0,
            optim_target: str = 'power',  # 'energy' or 'power'factory
    ):
        self.llm_engine = llm_engine
        self.vllm_config = vllm_config

        self.q: SimpleQueue = get_mp_context().SimpleQueue()
        self.server = _MPNvmlFreqModulatorServer(vllm_config,
                                                 freq_choices,
                                                 self.q,
                                                 log_dir=log_dir,
                                                 mod_interval=mod_interval,
                                                 tbt_sla=tbt_sla,
                                                 ttft_sla=ttft_sla,
                                                 optim_target=optim_target)
        self.server_process: Process = get_mp_context().Process(
            target=self.server.run)
        self.server_process.start()
        logger.info('_MPNvmlFreqModulatorServer process started.')

    def step(self, stats: Optional[Stats]) -> None:
        if stats:
            msg = self.build_msg(stats)
            msg_encoded = msgspec.msgpack.encode(msg)
            self.q.put(msg_encoded)

    def close(self):
        self.q.put(None)
        self.server_process.join()
        logger.info('_MPNvmlFreqModulatorServer process terminated.')

    @staticmethod
    def build_msg(stats: Stats) -> FreqModMsg:
        return FreqModMsg(
            stats.now,
            stats.running_queue_num_tokens_per_req,
            stats.wait_queue_num_prefill_tokens_per_req,
            stats.wait_queue_num_processed_tokens_per_req,
            stats.wait_queue_waiting_time_per_req,
            stats.gpu_cache_usage_sys,
            stats.num_precomputed_tokens_per_req_iter,
        )


class _MPNvmlFreqModulatorServer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        freq_choices: list[int],
        q: SimpleQueue,
        log_dir: Path,
        optim_target: str,
        mod_interval: int,
        tbt_sla: float,
        ttft_sla: float,
        future_window: int = 4,
        mem_util_ceiling: float = 0.9,
    ):
        self.vllm_config = vllm_config
        self.freq_choices = freq_choices
        self.q = q
        self.log_dir = log_dir

        self.future_windows = future_window
        self.mod_interval = mod_interval
        self.tbt_sla = tbt_sla
        self.ttft_sla = ttft_sla
        self.mem_util_ceiling = mem_util_ceiling
        self.optim_target = optim_target

        model_name = vllm_config.model_config.model.split('/')[-1]
        combo_name = f'{get_gpu_name()}_{model_name}'
        self.latency_model_dir = (PATH_TO_MODELS / 'latency_model' /
                                  combo_name)
        self.power_model_dir = (PATH_TO_MODELS / 'power_model' / combo_name)

        self.power_model: Booster
        self.latency_model_prefill: Booster
        self.latency_model_decode: Booster
        self.latency_model_hybrid: Booster

    def _load_models(self):
        self.power_model = Booster(model_file=self.power_model_dir /
                                   'power_model_all_grid-search.txt')
        self.latency_model_prefill = Booster(
            model_file=self.latency_model_dir /
            'latency_model_prefill-only_gdbt_grid-search.txt')
        self.latency_model_decode = Booster(
            model_file=self.latency_model_dir /
            'latency_model_decode-only_gdbt_grid-search.txt')
        self.latency_model_hybrid = Booster(
            model_file=self.latency_model_dir /
            'latency_model_hybrid_gdbt_grid-search.txt')

    def run(self):
        # Load models here rather than in __init__() so that we don't pass the
        # loaded models across processes
        self._load_models()

        # Column `now` used as key column to join with `perf_metrics.csv`
        csv_writer = CSVWriter(col_names=[
            'now', 'mpc_start', 'freq_mod_start', 'freq_mod_end',
            'target_freq', 'batch_lat', 'cpu_overhead'
        ],
                               filename=self.log_dir / 'freq_mod_log.csv')

        for step_id in count():
            msg_encoded = self.q.get()
            if msg_encoded is None:
                break
            if step_id % self.mod_interval > 0:
                continue
            mpc_start = time.perf_counter()
            msg: FreqModMsg = msgspec.msgpack.decode(msg_encoded,
                                                     type=FreqModMsg)
            logger.debug('freq_mod_msg: %s', msg)

            future_states, prefill_cycles = self.get_future_states(
                msg, self.future_windows)

            num_waiting_reqs = len(msg.wait_queue_num_prefill_tokens_per_req)
            # Smaller if not all requests are prefilled in `future_windows`
            assert len(prefill_cycles) <= num_waiting_reqs

            selected_freq, pred_batch_lat, pred_overhead = (
                self._get_next_freq_dp(msg, future_states, prefill_cycles))
            if msg.gpu_cache_usage_sys >= self.mem_util_ceiling:
                selected_freq = max(self.freq_choices)

            freq_mod_start = time.perf_counter()
            nvml_set_freq(selected_freq)
            freq_mod_end = time.perf_counter()
            csv_writer.add_row([
                msg.now,
                mpc_start,
                freq_mod_start,
                freq_mod_end,
                selected_freq,
                pred_batch_lat,
                pred_overhead,
            ])

        csv_writer.close()

    def _get_next_freq_dp(self, freq_mod_msg: FreqModMsg,
                          future_states: list[FutureState], prefill_cycles):
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
                freq_mod_msg.
                wait_queue_waiting_time_per_req[:len(prefill_cycles)])[:, None]
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
                if self.optim_target == 'energy':
                    selected_freq_ids = candidates[np.argmin(total_energy)]
                elif self.optim_target == 'power':
                    lat_per_batch = lat_mat[np.arange(max_future_vision)[:,
                                                                         None],
                                            candidates.T]
                    total_lat = np.sum(lat_per_batch, axis=0)
                    total_power = total_energy / total_lat
                    selected_freq_ids = candidates[np.argmin(total_power)]
                else:
                    raise NotImplementedError(self.optim_target)
            else:
                break

        selected_freq = max([
            freq_choices_desc[selected_freq_ids[i]]
            for i in range(self.mod_interval)
        ])
        predicted_overhead = self.get_cpu_overhead_us(
            future_states[0].num_decodes) / 1e6
        predicted_batch_lat = lat_mat_list[0][
            selected_freq_ids[0]] - predicted_overhead

        return selected_freq, predicted_batch_lat, predicted_overhead

    def get_future_states(self, msg: FreqModMsg,
                          future_window: int) -> tuple[list, list]:
        """
        Get the future observation for the given index. The future observation
        is a list of observations for the next `future_windows` batches.
        Assumptions:
            - prefills are (poorly) chunked now
            - no requests reach EOS during future calculations
            - prefill budget hard coded to 1024 (for now)
        """
        # A list that tells you for each request in the wait queue
        # how many iterations it will take to get the first token
        prefill_cycles = []

        # Construct a dummy wait queue to simulate future chunked prefills
        num_prefill_tokens = msg.wait_queue_num_prefill_tokens_per_req
        num_processed_tokens = msg.wait_queue_num_processed_tokens_per_req
        decode_precomputed_tokens = msg.num_precomputed_tokens_per_req_iter
        # list of (total tokens, processed tokens, remaining tokens)
        dummy_wait_queue = [
            (m, n, m - n)
            for m, n in zip(num_prefill_tokens, num_processed_tokens)
        ]

        num_decodes = len(msg.running_queue_num_tokens_per_req)
        if num_decodes > 0:
            decode_len_sum = sum(msg.running_queue_num_tokens_per_req)
            decode_len_max = max(msg.running_queue_num_tokens_per_req)
            decode_len_std = np.std(
                msg.running_queue_num_tokens_per_req).item()
        else:
            decode_len_sum = 0
            decode_len_max = 0
            decode_len_std = 0

        future_states = []
        for i in range(future_window):
            # Chunked prefill logic
            budget_left = 1024
            prefills = []
            prefill_precomputed_tokens = []
            precomputed_token = []
            chunked = False
            while budget_left > 0 and len(dummy_wait_queue) > 0:
                num_tokens = min(budget_left, dummy_wait_queue[0][2])
                prefills.append(num_tokens)
                prefill_precomputed_tokens.append(dummy_wait_queue[0][1])

                total_tokens, \
                processed_tokens, \
                remaining_tokens = dummy_wait_queue[0]
                budget_left -= num_tokens
                processed_tokens += num_tokens  # Update processed tokens
                remaining_tokens -= num_tokens  # Update remaining tokens
                dummy_wait_queue[0] = (total_tokens, processed_tokens,
                                       remaining_tokens)  # Update tuple
                chunked = True
                if dummy_wait_queue[0][2] == 0:
                    dummy_wait_queue.pop(0)
                    prefill_cycles.append(i + 1)
                    chunked = False

            num_prefills = len(prefills)
            if num_prefills > 0:
                prefill_len_sum = np.sum(prefills).item()
                prefill_len_max = np.max(prefills, initial=prefills[0])
                prefill_len_std = np.std(prefills).item()
            else:
                prefill_len_sum = 0
                prefill_len_max = 0
                prefill_len_std = 0.0

            precomputed_token = prefill_precomputed_tokens + \
                                decode_precomputed_tokens
            if len(precomputed_token) > 0:
                computed_prefill_len_sum = np.sum(precomputed_token).item()
                computed_prefill_len_max = np.max(precomputed_token,
                                                  initial=precomputed_token[0])
                computed_prefill_len_std = np.std(precomputed_token).item()
            else:
                computed_prefill_len_sum = 0
                computed_prefill_len_max = 0
                computed_prefill_len_std = 0.0

            future_states.append(
                FutureState(
                    num_prefills,
                    prefill_len_sum,
                    prefill_len_max,
                    prefill_len_std,
                    num_decodes,
                    decode_len_sum,
                    decode_len_max,
                    decode_len_std,
                    computed_prefill_len_sum,
                    computed_prefill_len_std,
                    computed_prefill_len_max,
                ))

            decode_precomputed_tokens = [
                x + 1 for x in decode_precomputed_tokens
            ]
            # Update decode statistics
            if chunked and num_prefills > 1:
                prefills_wo_len_sum = np.sum(prefills[:-1]).item()
                prefills_wo_len_max = np.max(prefills[:-1],
                                             initial=prefills[0])
                prefills_wo_len_std = np.std(prefills[:-1]).item()

                decode_len_max = max(decode_len_max, prefills_wo_len_max) + 1
                decode_len_std = np.sqrt(
                    (num_decodes * decode_len_std**2 +
                     (num_prefills - 1) * prefills_wo_len_std**2) /
                    (num_decodes + num_prefills + 1e-6))
                decode_len_sum += num_decodes + prefills_wo_len_sum
                num_decodes += num_prefills - 1
                decode_precomputed_tokens = prefill_precomputed_tokens[:-1] + \
                                            decode_precomputed_tokens
            elif chunked:
                decode_len_max = decode_len_max + 1
                decode_len_std = decode_len_std
                decode_len_sum += num_decodes
            else:
                decode_len_max = max(decode_len_max, prefill_len_max) + 1
                decode_len_std = np.sqrt((num_decodes * decode_len_std**2 +
                                          num_prefills * prefill_len_std**2) /
                                         (num_decodes + num_prefills + 1e-6))
                # all requests progress by 1 and prefills are added to decodes
                decode_len_sum += num_decodes + prefill_len_sum + num_prefills
                num_decodes += num_prefills
                decode_precomputed_tokens = prefill_precomputed_tokens + \
                                            decode_precomputed_tokens

        return future_states, prefill_cycles

    def predict_latencies(self, states: FutureState,
                          freq_choices) -> list[float]:
        """
        Predict latency of the upcoming batch for each freq in `freq_choices`.
        """
        num_prefills = states.num_prefills
        prefill_len_sum = states.prefill_len_sum
        prefill_len_std = states.prefill_len_std
        prefill_len_max = states.prefill_len_max
        num_decodes = states.num_decodes
        decode_len_sum = states.decode_len_sum
        decode_len_std = states.decode_len_std
        decode_len_max = states.decode_len_max
        running_queue_len = states.num_decodes
        computed_prefill_len_sum = states.computed_prefill_len_sum
        computed_prefill_len_std = states.computed_prefill_len_std
        computed_prefill_len_max = states.computed_prefill_len_max

        # hybrid model
        if num_prefills > 0 and num_decodes > 0:
            latency_model = self.latency_model_hybrid
            input = np.array([
                num_prefills, prefill_len_sum, prefill_len_std,
                prefill_len_max, computed_prefill_len_sum,
                computed_prefill_len_std, computed_prefill_len_max,
                num_decodes, decode_len_sum, decode_len_std, decode_len_max
            ])
        # prefill model
        elif num_prefills > 0:
            latency_model = self.latency_model_prefill
            input = np.array([
                num_prefills,
                prefill_len_sum,
                prefill_len_std,
                prefill_len_max,
                computed_prefill_len_sum,
                computed_prefill_len_std,
                computed_prefill_len_max,
            ])
        # decode model
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

    def predict_powers_future_states(self, future_states: list[FutureState],
                                     freq_choices) -> list[list[float]]:
        """
        Predict power of the all batches for each freq in `freq_choices`.
        """
        inputs = []
        for states in future_states:
            num_prefills = states.num_prefills
            prefill_len_sum = states.prefill_len_sum
            prefill_len_std = states.prefill_len_std
            prefill_len_max = states.prefill_len_max
            num_decodes = states.num_decodes
            decode_len_sum = states.decode_len_sum
            decode_len_std = states.decode_len_std
            decode_len_max = states.decode_len_max
            computed_prefill_len_sum = states.computed_prefill_len_sum
            computed_prefill_len_std = states.computed_prefill_len_std
            computed_prefill_len_max = states.computed_prefill_len_max

            input = np.array([
                num_prefills, prefill_len_sum, prefill_len_std,
                prefill_len_max, computed_prefill_len_sum,
                computed_prefill_len_std, computed_prefill_len_max,
                num_decodes, decode_len_sum, decode_len_std, decode_len_max
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

    @staticmethod
    def get_cpu_overhead_us(running_queue_len: int) -> float:
        cpu_overhead_us = (0.1502 * running_queue_len**2 +
                           87.8455 * running_queue_len - 894.6241)
        cpu_overhead_us = max(0.0, cpu_overhead_us)
        return cpu_overhead_us


if __name__ == '__main__':
    q: SimpleQueue = SimpleQueue()
    vllm_config = VllmConfig()
    vllm_config.model_config = ModelConfig(
        model='meta-llama/Llama-3.1-8B-Instruct',
        # Assign arbitrary values to remaining mandatory params
        task='draft',
        tokenizer='',
        tokenizer_mode='auto',
        trust_remote_code=False,
        dtype='float32',
        seed=0,
    )
    freq_choices = get_preselected_freq(get_gpu_name())
    s = _MPNvmlFreqModulatorServer(freq_choices=freq_choices,
                                   vllm_config=vllm_config,
                                   q=q,
                                   log_dir=Path('./logs'),
                                   optim_target='energy',
                                   mod_interval=1,
                                   tbt_sla=0.25,
                                   ttft_sla=1.0)
    msg = FreqModMsg(
        now=0.0,
        running_queue_num_tokens_per_req=[1074],
        wait_queue_num_prefill_tokens_per_req=[2050, 789],
        wait_queue_num_processed_tokens_per_req=[0, 0],
        wait_queue_waiting_time_per_req=[0, 0],
        gpu_cache_usage_sys=0.1,
        num_precomputed_tokens_per_req_iter=[25],
    )
    for _ in range(10):
        q.put(msgspec.msgpack.encode(msg))
    s.run()
