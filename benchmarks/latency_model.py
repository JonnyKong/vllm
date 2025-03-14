# SPDX-License-Identifier: Apache-2.0
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_batch import BenchmarkBatchParam
from benchmark_batch_driver import gen_power_profiling_args
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm


def main(batch_type: str):
    assert batch_type in ['prefill-only', 'decode-only', 'hybrid']

    X, Y, freqs = load_data(batch_type)
    print(f"Loaded {len(X)} samples.")
    X_train, X_test, Y_train, Y_test, freqs_train, freqs_test = \
        train_test_split(X, Y, freqs, test_size=0.1, random_state=0)

    # Train Gradient Boosting Regressor
    model = TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(n_estimators=100, random_state=0),
        transformer=FunctionTransformer(np.log, np.exp)  # Log transform Y
    )
    model.fit(X_train, Y_train)
    with open(f'latency_model_{batch_type}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Predict on test set
    Y_pred = model.predict(X_test)

    # Compute absolute relative error
    abs_rel_error = np.abs((Y_test - Y_pred) / Y_test)
    rel_error = (Y_test - Y_pred) / Y_test

    # Store errors in a pandas DataFrame and analyze grouped by frequency
    df_errors = pd.DataFrame({
        'Frequency': freqs_test,
        'Absolute Relative Error': abs_rel_error,
        'Relative Error': rel_error
    })

    grouped_errors = df_errors.groupby(
        'Frequency')['Absolute Relative Error'].mean()
    grouped_relative_errors_mean = df_errors.groupby(
        'Frequency')['Relative Error'].mean()
    grouped_relative_erros_std = df_errors.groupby(
        'Frequency')['Relative Error'].std()
    sample_counts = df_errors.groupby('Frequency').size()

    for freq, mean_error in grouped_errors.items():
        count = sample_counts[freq]
        rel_error_mean = grouped_relative_errors_mean[freq]
        rel_error_std = grouped_relative_erros_std[freq]
        print(f"Frequency {freq:.2f} "
              f"MHz (Samples: {count}) "
              f"MAE: {mean_error:.4f}, "
              f"Relative Error Mean: {rel_error_mean:.4f}, "
              f"Relative Error Std: {rel_error_std:.4f}")
    print(
        f"Overall Mean Absolute Relative Error: {np.mean(abs_rel_error):.4f}")


def load_data(batch_type: str):
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
        if not perf_path.exists():
            print(f"Skipping {args.log_dir}, missing required files.")
            continue
        # print(f"Loading {args.log_dir} ...")
        try:
            feat = get_feat(args)
            df_perf = pd.read_csv(perf_path)
            latency = (df_perf['pp_rank_0_end'] -
                       df_perf['pp_rank_0_start']).mean()

            if batch_type == 'hybrid':
                X.append(feat[:])
            elif batch_type == 'prefill-only':
                X.append(feat[:5])
            else:  # batch_type == 'decode-only'
                X.append(feat[[0, 5, 6, 7, 8]])
            freqs.append(feat[0])  # Store frequency value
            Y.append(latency)
        except Exception as e:
            print(f"Error processing {args.log_dir}: {e}")
            continue

    return np.array(X), np.array(Y), np.array(freqs)


def get_feat(p: BenchmarkBatchParam) -> np.ndarray:
    freq = float(p.gpu_freq_mhz)

    prefill_batch_size = len(p.prefill_input_lens)
    if prefill_batch_size == 0:
        prefill_len_sum = prefill_len_std = prefill_len_max = 0
    else:
        prefill_len_sum = np.sum(p.prefill_input_lens)
        prefill_len_std = np.std(p.prefill_input_lens)
        prefill_len_max = np.max(p.prefill_input_lens)

    decode_batch_size = len(p.decode_input_lens)
    if len(p.decode_input_lens) == 0:
        decode_len_sum = decode_len_std = decode_len_max = 0
    else:
        decode_len_sum = np.sum(p.decode_input_lens)
        decode_len_std = np.std(p.decode_input_lens)
        decode_len_max = np.max(p.decode_input_lens)

    return np.array([
        freq,
        prefill_batch_size,
        prefill_len_sum,
        prefill_len_std,
        prefill_len_max,
        decode_batch_size,
        decode_len_sum,
        decode_len_std,
        decode_len_max,
    ]).astype(np.float32)


if __name__ == '__main__':
    for batch_type in ['prefill-only', 'decode-only', 'hybrid']:
        main(batch_type)
