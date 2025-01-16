import csv
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
        self.power_logs = []  # Stores power readings with timestamps
        self.stop_monitoring = False

    def monitor_power(self):
        pynvml.nvmlInit()
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)
            ]

            column_names = ["Timestamp"] \
                    + [f"GPU_{i}" for i in range(gpu_count)]
            os.remove(self.csv_filename)
            with open(self.csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)

            logger.info('Monitoring power usage for %d GPUs...', gpu_count)
            last_log_time = time.perf_counter()

            while not self.stop_monitoring:
                timestamp = time.perf_counter()
                power_readings = [
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    for handle in handles
                ]  # Convert mW to W
                self.power_logs.append([timestamp] + power_readings)

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
        if self.power_logs:
            with open(self.csv_filename, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.power_logs)
            logger.debug('Appended %d log entries to %s', len(self.power_logs),
                         self.csv_filename)
            self.power_logs = []  # Clear logs after writing


def start_nvml_monitor(interval=0.1,
                       csv_filename="power_log.csv",
                       log_interval=1):
    monitor = NvmlPowerMonitor(interval=interval,
                               csv_filename=csv_filename,
                               log_interval=log_interval)
    monitor.monitor_power()
