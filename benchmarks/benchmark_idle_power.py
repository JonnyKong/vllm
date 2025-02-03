import time
from pathlib import Path
from typing import List

import pandas as pd
from benchmark_batch_driver import uniform_sample_sorted
from benchmark_utils import get_gpu_name, get_result_root

from vllm.logger import init_logger
from vllm.platforms.nvml_power_monitor import measure_power
from vllm.platforms.nvml_utils import nvml_get_available_freq, nvml_set_freq

logger = init_logger(__name__)


def measure_idle_power(freq_arr: List[int],
                       output_dir: Path,
                       duration: int = 3):
    """
    Measures GPU power consumption while idle at different frequencies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for freq in freq_arr:
        csv_filename = output_dir / f'power_{freq}MHz.csv'

        logger.info('Measuring power at %d MHz, logging to %s', freq,
                    csv_filename)

        with nvml_set_freq(freq), \
            measure_power(csv_filename,
                          interval=0.1,
                          log_interval=1,
                          enable_mem_freq_meas=True):
            time.sleep(duration)  # Keep the GPU idle while measuring

        logger.info('Completed measurement for %d MHz.', freq)

    logger.info('All measurements completed.')
    calculate_mean_power(output_dir)


def calculate_mean_power(output_dir: Path):
    """
    Calculates the mean power consumption and optional frequency values for
    each GPU frequency and saves results to a CSV.
    """
    results = []

    for csv_file in output_dir.glob('power_*MHz.csv'):
        freq = int(csv_file.stem.split('_')[1][:-3])
        df = pd.read_csv(csv_file)
        mean_power = df['GPU_0_power_w'].mean()

        mean_gpu_freq = df['GPU_0_freq_mhz'].mean(
        ) if 'GPU_0_freq_mhz' in df else None
        mean_mem_freq = df['GPU_0_mem_freq_mhz'].mean(
        ) if 'GPU_0_mem_freq_mhz' in df else None

        results.append((freq, mean_power, mean_gpu_freq, mean_mem_freq))

    results.sort()

    results_df = pd.DataFrame(
        results, columns=['freq', 'power', 'gpu_freq', 'mem_freq'])
    results_csv = output_dir / 'mean_power_results.csv'
    results_df.to_csv(results_csv, index=False)

    logger.info('Mean power consumption results saved to %s', results_csv)


if __name__ == '__main__':
    freq_arr = uniform_sample_sorted(nvml_get_available_freq(), k=16)
    output_dir = get_result_root() / 'idle_power' / get_gpu_name()
    measure_idle_power(freq_arr, output_dir)
