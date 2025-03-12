# SPDX-License-Identifier: Apache-2.0
import json
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def main():

    with open(
            "profiles/decode-only/profile_L4_Llama3-8B_decode-only.json") as f:
        profile = json.load(f)

    # Unomment for decode only
    X = [
        np.array([
            d["frequency"], d["decode_batch_size"], d["sum_decode_len"],
            d["max_decode_len"], d["std_decode_len"]
        ]) for d in profile
    ]

    # Uncomment for prefill only
    # X = [
    #     np.array([
    #         d["frequency"], d["prefill_batch_size"], d["sum_prefill_len"],
    #         d["max_prefill_len"], d["std_prefill_len"]
    #     ]) for d in profile
    # ]

    # Uncomment for hybrid
    # X = [
    #     np.array([
    #         d["frequency"], d["prefill_batch_size"], d["decode_batch_size"],
    #         d["sum_decode_len"], d["sum_prefill_len"], d["max_decode_len"],
    #         d["max_prefill_len"], d["std_decode_len"], d["std_prefill_len"]
    #     ]) for d in profile
    # ]

    Y = [np.mean(d["latencies"][:]) for d in profile]

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.1, random_state=0)

    model = GradientBoostingRegressor(random_state=0).fit(X_train, Y_train)

    errors = []
    differences = []
    count = 0

    for i in range(len(X_test)):
        if (True):  # Insert condition here
            # Ex: X_test[i][0] >= 1800 and X_test[i][0] < 2050
            # will only test frequencies between 1800 MHz and 2050 MHz
            predicted_latency = model.predict([X_test[i]])
            errors.append(abs(predicted_latency[0] - Y_test[i]) / Y_test[i])
            differences.append((predicted_latency[0] - Y_test[i]) * 1000)
            count += 1

    errors = sorted(errors)
    differences = sorted(differences)
    cum_error = sum(errors)

    avg_error = cum_error / count

    print("Samples:", count)
    print("Average error:", avg_error)
    print("25th percentile difference:",
          differences[int(len(differences) * 0.25)])
    print("75th percentile difference:",
          differences[int(len(differences) * 0.75)])

    with open('batch_latency_predictor_L4-LLama3-8B_decode-only.pkl',
              'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
