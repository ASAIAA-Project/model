import numpy as np


def accuracy_bi(y_pred, y):
    y_pred = ((y_pred * np.arange(1, 11)).sum(axis=1).round() > 5)
    y = ((y * np.arange(1, 11)).sum(axis=1).round() > 5)
    return (y_pred == y).sum() / len(y)


def accuracy_ten(y_pred, y):
    y_pred = (y_pred * np.arange(1, 11)).sum(axis=1).round()
    y = (y * np.arange(1, 11)).sum(axis=1).round()
    return (y_pred == y).sum() / len(y)


def accuracy_close(y_pred, y):
    y_pred = (y_pred * np.arange(1, 11)).sum(axis=1)
    y = (y * np.arange(1, 11)).sum(axis=1)
    return (abs(y_pred - y) < 1).sum() / len(y)
