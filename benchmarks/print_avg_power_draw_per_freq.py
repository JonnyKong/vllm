# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pandas as pd
from benchmark_batch_driver import gen_from_trace
from power_and_latency_model import compute_average_power
from tqdm import tqdm


def print_avg_power_draw_per_freq(gpu: str, model: str):
    samples = gen_from_trace(
        gpu,
        model,
        skip_existing=False,
    )
    df = []
    for args in tqdm(samples):
        perf_path = Path(args.log_dir) / 'perf_metric.csv'
        power_path = Path(args.log_dir) / 'power_log.csv'
        try:
            df_perf = pd.read_csv(perf_path)
            df_power = pd.read_csv(power_path)
            power = compute_average_power(df_perf, df_power)
        except (KeyError, pd.errors.EmptyDataError):
            continue
        df.append({
            'freq': args.gpu_freq_mhz,
            'power': power,
        })
    df = pd.DataFrame(df)
    print(df.groupby(['freq']).mean())


if __name__ == '__main__':
    gpu_model_combos = [
        # ['T4', 'phi-2'],
        ['A40', 'Llama-3.1-8B-Instruct'],
        ['A100-SXM4-80GB', 'gemma-2-27b-it'],
        # ['H100-80GB-HBM3', 'gemma-2-27b-it'],
        # ['A100-SXM4-80GB', 'Llama-3.1-70B-Instruct'],
    ]
    for gpu, model in gpu_model_combos:
        print('GPU')
        print_avg_power_draw_per_freq(gpu, model)
