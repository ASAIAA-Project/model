import numpy as np


def accuracy(y_pred, y):
    y_pred = np.round(y_pred[:, 0])
    y = np.round(y[:, 0])
    return (y_pred == y).sum() / len(y)
