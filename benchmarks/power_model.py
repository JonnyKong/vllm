# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_batch_driver import yield_benchmark_power_profiling


def main():
    for args in yield_benchmark_power_profiling(tp=1, pp=1):
        df_perf = pd.read_csv(Path(args.log_dir) / 'perf_metric.csv')
        df_power = pd.read_csv(Path(args.log_dir) / 'power_log.csv')
        power = compute_average_power(df_perf, df_power)
        print(power)


def filter_power_changes(df_power):
    """
    Filters the input dataframe to retain only rows where the GPU power changes
    or at least 110 milliseconds have passed since the last kept row.
    """
    df_power = df_power.sort_values(by='Timestamp').reset_index(drop=True)

    indices_to_keep = []
    last_kept_index = 0
    indices_to_keep.append(last_kept_index)

    for i in range(1, len(df_power)):
        time_diff = df_power.loc[i,
                                 'Timestamp'] - df_power.loc[last_kept_index,
                                                             'Timestamp']
        power_changed = df_power.loc[i, 'GPU_0_power_w'] != df_power.loc[
            last_kept_index, 'GPU_0_power_w']

        if power_changed or time_diff >= 0.110:
            indices_to_keep.append(i)
            last_kept_index = i

    return df_power.loc[indices_to_keep].reset_index(drop=True)


def compute_average_power(df_perf, df_power) -> float:
    """
    Computes the average power per batch based on the following rules:
    
    1. First, filter the power readings to only include rows where the GPU
    power value changes. If the time interval between readings deviates from
    100ms by more than ±20%, a warning is printed.
    
    2. Determine whether all batch latencies are greater than 200ms. The
    latency of a batch is calculated as: `pp_rank_0_idle - pp_rank_0_start`.
    
    3. If all batches have latencies greater than 200ms, compute the average
    power per batch by:
        - Extracting all power readings recorded within the batch duration from
          `pp_rank_0_start` to `pp_rank_0_idle`.
        - Ignoring the first power reading of each batch.
        - Averaging the remaining power readings for each batch.
        - Taking the average of these per-batch power values to get the final
          result.
    
    4. If any batch has a latency shorter than 200ms:
        - Consider all power readings from the start of the first batch to the
          end of the last batch.
        - Ignore the first power reading in the first batch.
        - Compute the average power from the remaining readings.
    """
    df_power = filter_power_changes(df_power)

    first_batch_start = df_perf['pp_rank_0_start'].min()
    last_batch_end = df_perf['pp_rank_0_idle'].max()

    all_batches_above_200ms = (df_perf['pp_rank_0_idle'] -
                               df_perf['pp_rank_0_start']).min() > 0.2

    if all_batches_above_200ms:
        batch_powers = []
        for _, batch in df_perf.iterrows():
            batch_start, batch_end = batch['pp_rank_0_start'], batch[
                'pp_rank_0_idle']
            power_readings = df_power[
                (df_power['Timestamp'] >= batch_start)
                & (df_power['Timestamp'] <= batch_end)]['GPU_0_power_w']
            if not power_readings.empty:
                batch_powers.append(
                    power_readings.iloc[1:].mean())  # Exclude first reading
        ret = float(np.nanmean(batch_powers))
    else:
        power_readings = df_power[
            (df_power['Timestamp'] >= first_batch_start)
            & (df_power['Timestamp'] <= last_batch_end)]['GPU_0_power_w']
        ret = power_readings.iloc[1:].mean()
    return ret


if __name__ == '__main__':
    main()
