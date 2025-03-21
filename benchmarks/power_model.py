# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from benchmark_batch import BenchmarkBatchParam
from benchmark_batch_driver import gen_power_profiling_args
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main(batch_type: Optional[str] = None):
    """
    `batch_type`: if None, train a single model for all 3 batch types.
    """
    if batch_type:
        assert batch_type in ['hybrid', 'prefill-only', 'decode-only']
    X, Y, freqs = load_data(batch_type)
    X_train, X_test, Y_train, Y_test, freqs_train, freqs_test = \
        train_test_split(X, Y, freqs, test_size=0.1, random_state=0)

    model = LGBMRegressor(random_state=0, n_jobs=1)
    model.fit(X_train, Y_train)
    if batch_type:
        model_name = f'power_model_{batch_type}.txt'
    else:
        model_name = 'power_model.txt'
    model.booster_.save_model(model_name)

    # Predict on test set
    Y_pred = model.predict(X_test)

    # Compute absolute relative error
    abs_rel_error = np.abs((Y_test - Y_pred) / Y_test)

    # Store errors in a pandas DataFrame and analyze grouped by frequency
    df_errors = pd.DataFrame({
        'Frequency': freqs_test,
        'Absolute Relative Error': abs_rel_error
    })

    grouped_errors = df_errors.groupby(
        'Frequency')['Absolute Relative Error'].mean()
    sample_counts = df_errors.groupby('Frequency').size()

    for freq, mean_error in grouped_errors.items():
        count = sample_counts[freq]
        print(
            f"Frequency {freq:.2f} MHz (Samples: {count}) MAE: {mean_error:.4f}"
        )
    print(
        f"Overall Mean Absolute Relative Error: {np.mean(abs_rel_error):.4f}")


def load_data(batch_type: Optional[str]):
    X = []
    Y = []
    freqs = []  # Store frequencies for grouping

    print('Loading data ...')
    for args in tqdm(
            gen_power_profiling_args(tp=1,
                                     pp=1,
                                     skip_existing=False,
                                     batch_type=batch_type)):
        perf_path = Path(args.log_dir) / 'perf_metric.csv'
        power_path = Path(args.log_dir) / 'power_log.csv'

        if not perf_path.exists() or not power_path.exists():
            print(f"Skipping {args.log_dir}, missing required files.")
            continue

        try:
            feat = get_feat(args, batch_type)
            df_perf = pd.read_csv(perf_path)
            df_power = pd.read_csv(power_path)
            power = compute_average_power(df_perf, df_power)
        except KeyError:
            print('Error loading: ', args.log_dir)
            continue
        assert not np.isnan(power)

        X.append(feat)
        freqs.append(feat[0])  # Store frequency value
        Y.append(power)

    return np.array(X), np.array(Y), np.array(freqs)


def get_feat(p: BenchmarkBatchParam, batch_type: Optional[str]) -> np.ndarray:
    ret = np.array([p.gpu_freq_mhz], dtype=np.float32)

    if not batch_type or batch_type == 'prefill-only':
        prefill_batch_size = len(p.prefill_input_lens)
        if prefill_batch_size > 0:
            prefill_len_sum = np.sum(p.prefill_input_lens)
            prefill_len_std = np.std(p.prefill_input_lens)
            prefill_len_max = np.max(p.prefill_input_lens)
        else:
            prefill_len_sum = 0.0
            prefill_len_std = 0.0
            prefill_len_max = 0.0
        ret = np.hstack([
            ret,
            np.array([
                prefill_batch_size,
                prefill_len_sum,
                prefill_len_std,
                prefill_len_max,
            ])
        ])

    if not batch_type or batch_type == 'decode-only':
        decode_batch_size = len(p.decode_input_lens)
        if decode_batch_size > 0:
            decode_len_sum = np.sum(p.decode_input_lens)
            decode_len_std = np.std(p.decode_input_lens)
            decode_len_max = np.max(p.decode_input_lens)
        else:
            decode_len_sum = 0.0
            decode_len_std = 0.0
            decode_len_max = 0.0
        ret = np.hstack([
            ret,
            np.array([
                decode_batch_size,
                decode_len_sum,
                decode_len_std,
                decode_len_max,
            ])
        ])
    return ret


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
        time_diff = (df_power.loc[i, 'Timestamp'] -
                     df_power.loc[last_kept_index, 'Timestamp'])
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


def load_data_to_df(batch_type: Optional[str]):
    """
    To human-readable format.
    """
    col_names: list[str] = ['freq']
    if not batch_type or batch_type == 'prefill-only':
        col_names.extend([
            'prefill_bs', 'prefill_len_sum', 'prefill_len_std',
            'prefill_len_max'
        ])
    if not batch_type or batch_type == 'decode-only':
        col_names.extend([
            'decode_bs', 'decode_len_sum', 'decode_len_std', 'decode_len_max'
        ])
    col_names.append('power')

    rows = []
    for args in tqdm(
            gen_power_profiling_args(tp=1,
                                     pp=1,
                                     skip_existing=False,
                                     batch_type=batch_type)):
        perf_path = Path(args.log_dir) / 'perf_metric.csv'
        power_path = Path(args.log_dir) / 'power_log.csv'

        if not perf_path.exists() or not power_path.exists():
            print(f"Skipping {args.log_dir}, missing required files.")
            continue

        feat = get_feat(args, batch_type)
        df_perf = pd.read_csv(perf_path)
        df_power = pd.read_csv(power_path)
        power = compute_average_power(df_perf, df_power)
        rows.append([*feat.tolist(), power])

    savename = f'{batch_type}.csv' if batch_type else 'all.csv'
    pd.DataFrame(rows, columns=col_names).to_csv(savename, index=False)


if __name__ == '__main__':
    for batch_type in [None, 'prefill-only', 'decode-only', 'hybrid']:
        load_data_to_df(batch_type)
        main(batch_type)
