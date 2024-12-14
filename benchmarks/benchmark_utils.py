# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = extra_info["tensor_parallel_size"]

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)


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
