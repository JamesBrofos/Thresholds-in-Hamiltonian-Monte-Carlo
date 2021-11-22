import os

import numpy as np
from scipy import io


def load_dataset(dataset: str='banana'):
    """Load a logistic regression dataset and a particular train-test fold. Here
    are the shapes of the various datasets as a tuple containing the number of
    observations and number of covariates (excluding the intercept).

    'banana'        | (5300, 3)
    'breast_cancer' | (277, 10)
    'diabetis'      | (768, 9)
    'flare_solar'   | (1066, 10)
    'heart'         | (270, 14)
    'image'         | (2310, 19)
    'german'        | (1000, 21)
    'ringnorm'      | (7400, 21)
    'splice'        | (3175, 61)
    'thyroid'       | (215, 6)
    'titanic'       | (2201, 4)
    'twonorm'       | (7400, 21)
    'waveform'      | (5000, 22)

    Args:
        dataset: The name of the dataset to load.

    Returns:
        x: The covariates.
        y: The binary targets.

    """
    fold = 0
    data = io.loadmat(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data', '13-benchmarks.mat'))[dataset][0][0]
    train_index, test_index = data[2][fold] - 1, data[3][fold] - 1
    x_train, x_test, y_train, y_test = (
        data[0][train_index],
        data[0][test_index],
        data[1][train_index].ravel(),
        data[1][test_index].ravel())
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
    y_train[y_train < 0.0] = 0.0
    y_test[y_test < 0.0] = 0.0
    x = np.vstack((x_train, x_test))
    y = np.append(y_train, y_test)
    return x, y
