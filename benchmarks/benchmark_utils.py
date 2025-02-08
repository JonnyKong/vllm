import os
from pathlib import Path

import torch


def get_gpu_name():
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found')
    return torch.cuda.get_device_name(0).replace('NVIDIA ', '')


def get_result_root() -> Path:
    if "GOOGLE_CLOUD_PROJECT" in os.environ:
        return Path.home() / 'energy_efficient_serving_results'
    else:
        return Path('/export2/kong102/energy_efficient_serving_results')


def uniform_sample_sorted(lst, k):
    """
    Selects `k` elements from the sorted input list as uniformly as possible,
    ensuring the first and last elements are included.
    """
    if k < 2 or k > len(lst):
        raise ValueError(
            "k must be at least 2 and at most the length of the list")
    lst = sorted(lst)
    step = (len(lst) - 1) / (k - 1)
    indices = sorted(set(round(i * step) for i in range(k)))
    return [lst[i] for i in indices]
