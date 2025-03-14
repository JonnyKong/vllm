# SPDX-License-Identifier: Apache-2.0
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_batch import BenchmarkBatchParam
from benchmark_batch_driver import gen_power_profiling_args
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    # model_type = 'decode'
    model_type = 'prefill'
    # model_type = 'hybrid'

    X, Y, freqs = load_data(model_type)
    X_train, X_test, Y_train, Y_test, freqs_train, freqs_test = \
        train_test_split(X, Y, freqs, test_size=0.1, random_state=0)

    # Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(random_state=0, n_estimators=100)
    model.fit(X_train, Y_train)
    with open(f'{model_type}_latency_model.pkl', 'wb') as f:
        pickle.dump(model, f)

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


def load_data(model_type: str):
    X = []
    Y = []
    freqs = []  # Store frequencies for grouping

    print('Loading data ...')
    for args in tqdm(gen_power_profiling_args(tp=1, pp=1,
                                              skip_existing=False)):
        if model_type not in args.log_dir:
            continue
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

            if model_type == 'hybrid':
                X.append(feat[:])
            elif model_type == 'prefill':
                X.append(feat[:5])
            else:  # model_type == 'decode'
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
        decode_batch_size = decode_len_sum = decode_len_std = decode_len_max = 0
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
    ],
                    dtype=np.float32)


if __name__ == '__main__':
    main()
