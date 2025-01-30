import contextlib
import csv
import multiprocessing
import os
import time

import pynvml

from vllm.logger import init_logger

logger = init_logger(__name__)


class NvmlPowerMonitor:

    def __init__(self, interval, csv_filename, log_interval):
        self.interval = interval
        self.csv_filename = csv_filename
        self.log_interval = log_interval
        self.logs = []  # Stores power and frequency readings with timestamps
        self.stop_monitoring = False

    def monitor_power_and_freq(self):
        pynvml.nvmlInit()
        try:
            # Get the GPUs specified by CUDA_VISIBLE_DEVICES
            cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is not None:
                visible_indices = [
                    int(x) for x in cuda_visible_devices.split(",")
                ]
            else:
                visible_indices = list(range(pynvml.nvmlDeviceGetCount()))
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in visible_indices
            ]

            column_names = ["Timestamp"]
            for i in range(len(handles)):
                column_names.append(f"GPU_{i}_power_w")
                column_names.append(f"GPU_{i}_freq_mhz")

            os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
            if os.path.exists(self.csv_filename):
                os.remove(self.csv_filename)
            with open(self.csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)

            logger.info('Monitoring power and frequency for %d GPUs...',
                        len(handles))
            last_log_time = time.perf_counter()

            while not self.stop_monitoring:
                timestamp = time.perf_counter()
                readings = [timestamp]
                for handle in handles:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(
                        handle) / 1000.0  # Convert mW to W
                    freq = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS)  # Frequency in MHz
                    readings.extend([power_usage, freq])
                self.logs.append(readings)

                # Periodically write logs to CSV
                if timestamp - last_log_time >= self.log_interval:
                    self._write_logs_to_csv()
                    last_log_time = timestamp

                time.sleep(self.interval)

            # Write remaining logs on exit
            self._write_logs_to_csv()

        finally:
            pynvml.nvmlShutdown()

    def _write_logs_to_csv(self):
        if self.logs:
            with open(self.csv_filename, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.logs)
            logger.debug('Appended %d log entries to %s', len(self.logs),
                         self.csv_filename)
            self.logs = []  # Clear logs after writing


def start_nvml_monitor(interval: float, csv_filename: str, log_interval=1):
    monitor = NvmlPowerMonitor(interval=interval,
                               csv_filename=csv_filename,
                               log_interval=log_interval)
    monitor.monitor_power_and_freq()


@contextlib.contextmanager
def measure_power(csv_filename, interval=0.1, log_interval=1):
    process = multiprocessing.Process(target=start_nvml_monitor,
                                      args=(interval, csv_filename,
                                            log_interval))
    process.start()
    try:
        yield
    finally:
        process.terminate()
        process.join()
        logger.info("Power monitoring process terminated.")
