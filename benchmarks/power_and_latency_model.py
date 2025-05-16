# SPDX-License-Identifier: Apache-2.0
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from benchmark_batch import BenchmarkBatchParam
from benchmark_batch_driver import gen_from_trace
from latency_and_power_model_sampler import get_cdf_data
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MLPRegressor(nn.Module):

    def __init__(self, input_dim, lr=1e-3, epochs=1000, batch_size=32):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in [64, 64]:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def fit(self, X, y, log_dir: Path, test_size=0.2, random_state=42):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor,
                                                       y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        writer = SummaryWriter(log_dir=log_dir)

        for epoch in tqdm(range(self.epochs)):
            self.train()
            epoch_train_loss = 0
            for xb, yb in train_loader:
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            inf_lat_arr = []
            self.eval()
            with torch.no_grad():
                start = time.time()
                val_pred = self.model(X_val_tensor)
                inf_lat_arr.append(time.time() - start)
                val_loss = self.loss_fn(val_pred, y_val_tensor).item()
            train_loss_avg = epoch_train_loss / len(train_loader)

            if epoch > 20:
                # Skip rampup region where high losses enlarge the figure ylim
                writer.add_scalar('Loss/Train', train_loss_avg, epoch)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Latency/Inference', np.mean(inf_lat_arr),
                                  epoch)

        writer.close()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return self.model(X_tensor).squeeze().numpy()

    def save_model(self, path):
        path = Path(path)
        torch.save(self.state_dict(), path)

    def load_model(self, path, input_dim):
        self.__init__(input_dim)  # reinitialize model
        self.load_state_dict(torch.load(path))


def main(gpu: str,
         model: str,
         model_type: str,
         freq_to_keep: Optional[int] = None,
         enable_grid_search: bool = False,
         include_precomputed: bool = False):
    assert model_type in ['gdbt', 'mlp']

    for batch_type in ['prefill-only', 'decode-only', 'hybrid', 'all']:
        X, Y, freqs, _, features_df = load_data(gpu, model, batch_type,
                                                freq_to_keep,
                                                include_precomputed)
        result_root = Path('latency_model') / f'{gpu}_{model}'
        latency_model(X, Y, freqs, batch_type, model_type, enable_grid_search,
                      features_df, result_root)

    X, Y, freqs, powers, features_df = load_data(gpu, model, 'all',
                                                 freq_to_keep,
                                                 include_precomputed)
    result_root = Path('power_model') / f'{gpu}_{model}'
    power_model(X, powers, freqs, enable_grid_search, result_root)


def latency_model(X, Y, freqs, batch_type, model_type, enable_grid_search,
                  featuers_df, result_root: Path):
    result_root.mkdir(parents=True, exist_ok=True)

    model_name = f'latency_model_{batch_type}_{model_type}'
    if enable_grid_search:
        model_name += '_grid-search'

    X_train, X_test, Y_train, Y_test, freqs_train, freqs_test = \
        train_test_split(X, Y, freqs, test_size=0.1, random_state=0)
    Y_train_log = np.log(Y_train)

    if model_type == 'gdbt':
        model = LGBMRegressor(objective='l2')
        if enable_grid_search:
            param_grid = {
                'num_leaves': [20, 31, 40],
                'learning_rate': [0.5, 0.1, 0.05],
                'n_estimators': [50, 100, 200, 400],
            }
            grid_search = GridSearchCV(model,
                                       param_grid,
                                       cv=3,
                                       scoring='neg_mean_squared_error')
            grid_search.fit(X_train, Y_train_log)
            model = grid_search.best_estimator_
            print('best params: ', grid_search.best_params_)
        else:
            model.fit(X_train, Y_train_log)
        model.booster_.save_model(result_root / f'{model_name}.txt')
    else:
        model = MLPRegressor(input_dim=len(X_train[0]))
        model.fit(X_train, Y_train_log,
                  Path('latency_model') / f'{model_name}_loss_curve.pdf')
        model.save_model(result_root / f'{model_name}.pt')

    y_preds = []
    # Predict on both training and testing set
    df_errors = pd.DataFrame()
    featuers_df['Absolute Relative Error'] = -100
    featuers_df['Relative Error'] = -100
    featuers_df['Error'] = -100
    for split in ['train', 'test']:
        if split == 'train':
            X, Y, freqs = X_train, Y_train, freqs_train
        else:
            X, Y, freqs = X_test, Y_test, freqs_test
        print(f'split={split}, shape={X.shape}')
        Y_pred_log = model.predict(X)
        Y_pred = np.exp(Y_pred_log)
        y_preds.append(Y_pred)
        # Compute absolute relative error

        abs_rel_error = np.abs((Y_pred - Y) / Y)
        rel_error = (Y_pred - Y) / Y
        error = Y_pred - Y

        # Store errors in a pandas DataFrame and analyze grouped by frequency
        df_errors = pd.concat([
            df_errors,
            pd.DataFrame({
                'Frequency': freqs,
                'Absolute Relative Error': abs_rel_error,
                'Relative Error': rel_error,
                'Error': error,
                'Split': split,
            })
        ])

    featuers_df.to_csv(result_root / f'features_{model_name}.csv', index=False)

    plot_pred_error_cdf(df_errors, result_root / f'loss_cdf_{model_name}.pdf')


def power_model(X, Y, freqs, enable_grid_search, result_root: Path):
    result_root.mkdir(parents=True, exist_ok=True)

    model_name = 'power_model_all'
    if enable_grid_search:
        model_name += '_grid-search'
    X_train, X_test, Y_train, Y_test, freqs_train, freqs_test = \
        train_test_split(X, Y, freqs, test_size=0.1, random_state=0)

    model = LGBMRegressor(random_state=0, n_jobs=1)
    model.fit(X_train, Y_train)
    model.booster_.save_model(result_root / f'{model_name}.txt')

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

    # Plot error CDF
    x, y = get_cdf_data(abs_rel_error)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    fig.tight_layout()
    plt.savefig(result_root / f'loss_cdf_{model_name}.pdf')

    # Save errors to a CSV for later plotting
    df_errors.to_csv(result_root / 'power_errors.csv', index=False)


def plot_pred_error_cdf(df_errors: pd.DataFrame, output_path: Path):
    error_names = ['Absolute Relative Error', 'Relative Error', 'Error']
    fig, axs = plt.subplots(2, len(error_names), figsize=(15, 10))

    # Average over all freqs
    for ax, error_name in zip(axs[0], error_names):
        for split in ['train', 'test']:
            df = df_errors[df_errors['Split'] == split]
            x, y = get_cdf_data(df[error_name])
            ax.plot(x, y, label=split)
        ax.set_title(f'{error_name}')
        ax.set_ylabel('CDF')
        ax.set_xlabel('Error (positive means over-pred)')
        ax.legend()
        ax.grid()

    # For each freq on test set
    for ax, error_name in zip(axs[1], error_names):
        for freq in [210, 360, 510, 825, 975, 1125, 1275, 1440, 1590, 1740]:
            df_errors_ = df_errors[(df_errors['Split'] == 'test')
                                   & (df_errors['Frequency'] == freq)]
            x, y = get_cdf_data(df_errors_[error_name])
            ax.plot(x, y, label=str(freq))
        ax.set_title(f'{error_name}, mean: {df_errors[error_name].mean():.4f}')
        ax.set_ylabel('CDF')
        ax.set_xlabel('Error (positive means over-pred)')
        ax.grid()
        ax.legend()

    fig.tight_layout()
    plt.savefig(output_path)


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


def load_data(gpu: str, model: str, batch_type: str,
              freq_to_keep: Optional[int], include_precomputed: False):
    X = []
    Y = []
    freqs = []  # Store frequencies for grouping
    powers = []
    if include_precomputed:
        features_df = pd.DataFrame(columns=[
            'path', 'freq', 'prefill_batch_size', 'prefill_len_sum',
            'prefill_len_std', 'prefill_len_max', 'computed_prefill_len_sum',
            'computed_prefill_len_std', 'computed_prefill_len_max',
            'decode_batch_size', 'decode_len_sum', 'decode_len_std',
            'decode_len_max', 'latency'
        ])
    else:
        features_df = pd.DataFrame(columns=[
            'path', 'freq', 'prefill_batch_size', 'prefill_len_sum',
            'prefill_len_std', 'prefill_len_max', 'decode_batch_size',
            'decode_len_sum', 'decode_len_std', 'decode_len_max', 'latency'
        ])

    print('Loading data ...')

    samples = gen_from_trace(
        gpu,
        model,
        skip_existing=False,
    )
    print(f"Total samples: {len(samples)}")
    skipped = 0
    for args in tqdm(samples):
        if not (args.get_batch_type() == batch_type or batch_type == 'all'):
            continue
        perf_path = Path(args.log_dir) / 'perf_metric.csv'
        if not perf_path.exists():
            # print(f"Skipping {args.log_dir}, missing required perf files.")
            skipped += 1
            continue
        power_path = Path(args.log_dir) / 'power_log.csv'
        if not power_path.exists():
            # print(f"Skipping {args.log_dir}, missing required power files.")
            skipped += 1
            continue

        try:
            feat = get_feat(args, include_precomputed)
            df_perf = pd.read_csv(perf_path)
            df_power = pd.read_csv(power_path)
            latency = (df_perf['pp_rank_0_end'] -
                       df_perf['pp_rank_0_start']).mean()
            if include_precomputed:
                features_df = pd.concat([
                    features_df,
                    pd.DataFrame(
                        {
                            'path': args.log_dir,
                            'freq': feat[0],
                            'prefill_batch_size': feat[1],
                            'prefill_len_sum': feat[2],
                            'prefill_len_std': feat[3],
                            'prefill_len_max': feat[4],
                            'computed_prefill_len_sum': feat[5],
                            'computed_prefill_len_std': feat[6],
                            'computed_prefill_len_max': feat[7],
                            'decode_batch_size': feat[8],
                            'decode_len_sum': feat[9],
                            'decode_len_std': feat[10],
                            'decode_len_max': feat[11],
                            'latency': latency
                        },
                        index=[0])
                ],
                                        ignore_index=True)
            else:
                features_df = pd.concat([
                    features_df,
                    pd.DataFrame(
                        {
                            'path': args.log_dir,
                            'freq': feat[0],
                            'prefill_batch_size': feat[1],
                            'prefill_len_sum': feat[2],
                            'prefill_len_std': feat[3],
                            'prefill_len_max': feat[4],
                            'decode_batch_size': feat[5],
                            'decode_len_sum': feat[6],
                            'decode_len_std': feat[7],
                            'decode_len_max': feat[8],
                            'latency': latency
                        },
                        index=[0])
                ],
                                        ignore_index=True)

            if freq_to_keep and feat[0] != freq_to_keep:
                continue

            # Compute power first, so on exception, X and Y are not appended,
            # keeping them of the same length
            power = compute_average_power(df_perf, df_power)
            if np.isnan(power):
                continue

            if (include_precomputed):
                if batch_type == 'prefill-only':
                    X.append(feat[:8])
                elif batch_type == 'decode-only':
                    X.append(feat[[0, 8, 9, 10, 11]])
                else:
                    X.append(feat[:])
            else:
                if batch_type == 'prefill-only':
                    X.append(feat[:5])
                elif batch_type == 'decode-only':
                    X.append(feat[[0, 5, 6, 7, 8]])
                else:
                    X.append(feat[:])

            powers.append(power)
            freqs.append(feat[0])  # Store frequency value
            Y.append(latency)

        except Exception:
            # print(f"Error processing {args.log_dir}: {e}")
            skipped += 1
            continue
    print(f"Samples for {batch_type}: {len(X)}")
    print(f"Skipped {skipped} samples due to missing files or errors.")
    return np.array(X), np.array(Y), np.array(freqs), np.array(
        powers), features_df


def get_feat(p: BenchmarkBatchParam, include_precomputed: False) -> np.ndarray:
    freq = float(p.gpu_freq_mhz)

    prefill_batch_size = len(p.prefill_input_lens)
    prefill_completed_batch_size = len(p.prefill_completed_input_lens)
    if prefill_batch_size == 0:
        prefill_len_sum = prefill_len_std = prefill_len_max = 0
        computed_prefill_len_sum = computed_prefill_len_max = 0
        computed_prefill_len_std = 0
    elif prefill_completed_batch_size == 0:
        prefill_len_sum = np.sum(p.prefill_input_lens)
        prefill_len_std = np.std(p.prefill_input_lens)
        prefill_len_max = np.max(p.prefill_input_lens)
        computed_prefill_len_sum = computed_prefill_len_max = 0
        computed_prefill_len_std = 0
    else:
        prefill_lens = np.array(p.prefill_input_lens)
        computed_prefill_lens = np.array(p.prefill_completed_input_lens)
        uncomputed_prefill_lens = prefill_lens - computed_prefill_lens

        prefill_len_sum = np.sum(uncomputed_prefill_lens)
        prefill_len_std = np.std(uncomputed_prefill_lens)
        prefill_len_max = np.max(uncomputed_prefill_lens)

        computed_prefill_len_sum = np.sum(computed_prefill_lens)
        computed_prefill_len_std = np.std(computed_prefill_lens)
        computed_prefill_len_max = np.max(computed_prefill_lens)

    decode_batch_size = len(p.decode_input_lens)
    if len(p.decode_input_lens) == 0:
        decode_len_sum = decode_len_std = decode_len_max = 0
    else:
        decode_len_sum = np.sum(p.decode_input_lens)
        decode_len_std = np.std(p.decode_input_lens)
        decode_len_max = np.max(p.decode_input_lens)

    if include_precomputed:
        ret = np.array([
            freq,
            prefill_batch_size,
            prefill_len_sum,
            prefill_len_std,
            prefill_len_max,
            computed_prefill_len_sum,
            computed_prefill_len_std,
            computed_prefill_len_max,
            decode_batch_size,
            decode_len_sum,
            decode_len_std,
            decode_len_max,
        ]).astype(np.float32)
    else:
        ret = np.array([
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

    return ret


if __name__ == '__main__':
    gpu_model_combos = [
        ['T4', 'phi-2'],
        ['A40', 'Llama-3.1-8B-Instruct'],
        ['A100-SXM4-80GB', 'gemma-2-27b-it'],
        ['H100-80GB-HBM3', 'gemma-2-27b-it'],
        ['A100-SXM4-80GB', 'Llama-3.1-70B-Instruct'],
    ]
    for gpu, model in gpu_model_combos:
        main(gpu,
             model,
             model_type='gdbt',
             enable_grid_search=True,
             include_precomputed=True)
