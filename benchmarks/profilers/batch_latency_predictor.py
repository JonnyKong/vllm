# SPDX-License-Identifier: Apache-2.0
import json
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def main():

    with open("profile.json") as f:
        profile = json.load(f)

    X = [np.array(list(d.values())[1:]) for d in profile]
    Y = [np.array(list(d.values())[0]) for d in profile]

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.15, random_state=0)

    model = GradientBoostingRegressor(random_state=0).fit(X_train, Y_train)

    # print("Model score:", model.score(X_train, Y_train))

    errors = []
    differences = []
    significance = [0.03, 0.05, 0.1]
    passed = [0] * len(significance)
    count = 0

    for i in range(len(X_test)):
        if (X_test[i][0] >= 1800 and X_test[i][0] < 2050):
            predicted_latency = model.predict([X_test[i]])
            errors.append(abs(predicted_latency[0] - Y_test[i]) / Y_test[i])
            differences.append((predicted_latency[0] - Y_test[i]) * 1000)
            for i, sig in enumerate(significance):
                if (abs(errors[-1]) < sig):
                    passed[i] += 1
            count += 1

    errors = sorted(errors)
    differences = sorted(differences)
    cum_error = sum(errors)

    avg_error = cum_error / count

    print("Samples:", count)
    print("Average error:", avg_error)

    with open("batch_latency_predictor_model.pkl", 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
