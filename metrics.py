import numpy as np


def accuracy_bi(y_pred, y):
    y_pred = np.round(y_pred[:, 0])
    y = np.round(y[:, 0])
    return (y_pred == y).sum() / len(y)


def accuracy_ten(y_pred, y):
    y_pred = (y_pred * np.arange(1, 11)).mean(axis=1).round()
    y = (y * np.arange(1, 11)).mean(axis=1).round()
    return (y_pred == y).sum() / len(y)
