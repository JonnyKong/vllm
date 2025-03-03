# SPDX-License-Identifier: Apache-2.0
from .nvml_freq_modulator import NvmlFreqModulator


class RuleBasedNvmlFreqModulator(NvmlFreqModulator):
    '''
    A rule-based implementation of NvmlFreqModulator. Adjusts the GPU frequency
    based on the fraction of active tasks in the scheduler.
    '''

    def __init__(self, llm_engine, interval_s: float) -> None:
        super().__init__(llm_engine, interval_s)

    frequency_table = {
        (0.0, 0.2): 1425,
        (0.2, 1.0): 1440,
    }

    def adjust(self) -> int:
        sys_stats = self.get_sys_stats()

        fraction = sys_stats['gpu_kv_cache_usage']

        for (low, high), freq in self.frequency_table.items():
            if low <= fraction < high:
                return freq

        return max(self.frequency_table.values())
