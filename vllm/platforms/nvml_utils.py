# SPDX-License-Identifier: Apache-2.0
import contextlib
import csv
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pynvml

from vllm.logger import init_logger

logger = init_logger(__name__)

_nvml_freq_active = False  # Prevent nested or concurrent frequency setting
_nvml_freq_lock = threading.Lock()  # Lock to prevent race conditions


def nvml_get_available_freq():
    """
    Returns a sorted list of available GPU clock frequencies at the highest
    memory clock setting.
    """
    pynvml.nvmlInit()
    handle = _get_gpu_handles()[0]

    memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
    highest_memory_clock = max(memory_clocks)

    return sorted(
        pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle,
                                                    highest_memory_clock))


@contextlib.contextmanager
def nvml_lock_freq(freq):
    """
    Context manager that temporarily locks GPU frequency for all GPUs in
    `CUDA_VISIBLE_DEVICES`. Prevents nested or concurrent usage across threads.
    """
    global _nvml_freq_active

    with _nvml_freq_lock:  # Ensure thread safety
        if _nvml_freq_active:
            raise RuntimeError(
                'nvml_lock_freq is already active in another thread!')

        _nvml_freq_active = True

    handles = _get_gpu_handles()
    try:
        for handle in handles:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
        logger.info('Locking GPU freq at %d MHz ...', freq)
        yield
    finally:
        for handle in handles:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        with _nvml_freq_lock:
            _nvml_freq_active = False
        logger.info('Resetting GPU freq ...')


def nvml_set_freq(freq):
    """
    Function that sets the GPU frequency for all GPUs in `CUDA_VISIBLE_DEVICES`.
    If the context manager `nvml_lock_freq` is active, raises an exception.
    """
    global _nvml_freq_active

    with _nvml_freq_lock:
        if _nvml_freq_active:
            raise RuntimeError(
                'Cannot set GPU frequency while nvml_lock_freq is active!')

    handles = _get_gpu_handles()
    for handle in handles:
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
    logger.info('Set GPU freq to %d MHz.', freq)


def _get_gpu_handles():
    pynvml.nvmlInit()
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        gpu_indices = [int(i) for i in cuda_visible_devices.split(',')]
    else:
        gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))
    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]


@contextmanager
def timeit(name='Unnamed code block'):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info('[%s] execution time: %f', name, elapsed_time)


class CSVWriter:

    def __init__(self, col_names: list[str], filename: Path):
        self.filename = str(filename)
        self.col_names = col_names
        self.header_written = False  # Track if the header has been written

        # ruff: noqa: SIM115
        self.file = open(self.filename, 'a', newline='', encoding='utf-8')
        # ruff: enable
        self.writer = csv.writer(self.file)

    def add_row(self, row: list):
        assert len(row) == len(self.col_names)
        if not self.header_written:
            self.writer.writerow(self.col_names)
            self.header_written = True
        self.writer.writerow(row)

    def close(self):
        self.file.close()
