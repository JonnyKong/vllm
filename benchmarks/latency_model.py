# SPDX-License-Identifier: Apache-2.0
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from benchmark_batch import BenchmarkBatchParam
from benchmark_batch_driver import gen_power_profiling_args
from latency_and_power_model_sampler import get_cdf_data
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MLPRegressor(nn.Module):

    def __init__(self, input_dim, lr=1e-3, epochs=2500, batch_size=32):
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


def main(batch_type: str, model_type: str):
    assert model_type in ['gdbt', 'mlp']
    assert batch_type in ['prefill-only', 'decode-only', 'hybrid']

    X, Y, freqs = load_data(batch_type)
    print(f"Loaded {len(X)} samples.")
    X_train, X_test, Y_train, Y_test, freqs_train, freqs_test = \
        train_test_split(X, Y, freqs, test_size=0.1, random_state=0)
    Y_train_log = np.log(Y_train)

    if model_type == 'gdbt':
        model = LGBMRegressor()
        model.fit(X_train, Y_train_log)
        model.booster_.save_model(
            Path('latency_model') / f'latency_model_{batch_type}.txt')
    else:
        model = MLPRegressor(input_dim=len(X_train[0]))
        model.fit(X_train, Y_train_log,
                  Path('latency_model') / f'mlp_loss_curve_{batch_type}')
        model.save_model(
            Path('latency_model') / f'latency_model_{batch_type}.pt')

    # Predict on test set
    Y_pred_log = model.predict(X_test)
    Y_pred = np.exp(Y_pred_log)

    # Compute absolute relative error
    abs_rel_error = np.abs((Y_pred - Y_test) / Y_test)
    rel_error = (Y_pred - Y_test) / Y_test
    error = Y_pred - Y_test

    # Store errors in a pandas DataFrame and analyze grouped by frequency
    df_errors = pd.DataFrame({
        'Frequency': freqs_test,
        'Absolute Relative Error': abs_rel_error,
        'Relative Error': rel_error,
        'Error': error,
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
    plot_pred_error_cdf(
        df_errors,
        Path('latency_model') / f'loss_cdf_{batch_type}_{model_type}.pdf')


def plot_pred_error_cdf(df_errors: pd.DataFrame, output_path: Path):
    error_names = ['Absolute Relative Error', 'Relative Error', 'Error']
    fig, axs = plt.subplots(2, len(error_names), figsize=(15, 10))

    # Average over all freqs
    for ax, error_name in zip(axs[0], error_names):
        x, y = get_cdf_data(df_errors[error_name])
        ax.plot(x, y, label=error_name)
        ax.set_title(f'{error_name}, mean: {df_errors[error_name].mean():.4f}')
        ax.set_ylabel('CDF')
        ax.set_xlabel('Error (positive means over-pred)')
        ax.grid()

    # For each freq
    for ax, error_name in zip(axs[1], error_names):
        for freq in [825, 975, 1125, 1275, 1440, 1590, 1740]:
            df_errors_ = df_errors[df_errors['Frequency'] == freq]
            x, y = get_cdf_data(df_errors_[error_name])
            ax.plot(x, y, label=str(freq))
        ax.set_title(f'{error_name}, mean: {df_errors[error_name].mean():.4f}')
        ax.set_ylabel('CDF')
        ax.set_xlabel('Error (positive means over-pred)')
        ax.grid()
        ax.legend()

    fig.tight_layout()
    plt.savefig(output_path)


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
        main(batch_type, 'mlp')
        main(batch_type, 'gdbt')
