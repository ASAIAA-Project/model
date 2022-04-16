import numpy as np


def accuracy(y, y_pred):
    y_pred = np.round(y_pred[:, 0])
    y = np.round(y[:, 0])
    return (y == y_pred).sum() / len(y)
