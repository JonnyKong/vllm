import contextlib
import os

import pynvml

from vllm.logger import init_logger

logger = init_logger(__name__)


def nvml_get_available_freq():
    """
    Returns a sorted list of available GPU clock frequencies at the highest
    memory clock setting.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
    highest_memory_clock = max(memory_clocks)

    return sorted(
        pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle,
                                                    highest_memory_clock))


@contextlib.contextmanager
def nvml_set_freq(freq):
    """
    Temporarily set and restore GPU frequency for all GPUs in
    `CUDA_VISIBLE_DEVICES`.
    """
    pynvml.nvmlInit()

    # Determine which GPUs are visible
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_indices = [int(i) for i in cuda_visible_devices.split(",")]
    else:
        gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))

    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]

    try:
        for handle in handles:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
        logger.info('Setting GPU freq to %d MHz ...', freq)
        yield
    finally:
        for handle in handles:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        logger.info('Resetting GPU freq ...')
