# SPDX-License-Identifier: Apache-2.0
from .nvml_freq_modulator import InProcNvmlFreqModulator


class RuleBasedNvmlFreqModulator(InProcNvmlFreqModulator):
    '''
    A rule-based implementation of NvmlFreqModulator. Adjusts the GPU frequency
    based on the fraction of active tasks in the scheduler.
    '''

    def __init__(self, llm_engine, interval_s: float) -> None:
        super().__init__(llm_engine, interval_s)

    frequency_table = {
        (0.0, 0.3): 1125,
        (0.3, 0.7): 1440,
        (0.7, 1.0): 1740,
    }

    def adjust(self) -> int:
        sys_stats = self.get_sys_stats()

        fraction = sys_stats['gpu_kv_cache_usage']

        for (low, high), freq in self.frequency_table.items():
            if low <= fraction < high:
                return freq

        return max(self.frequency_table.values())
