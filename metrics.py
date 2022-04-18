import numpy as np
import torch


def accuracy_bi(y_pred, y):
    y_pred = np.round(y_pred[:, 0])
    y = np.round(y[:, 0])
    return (y_pred == y).sum() / len(y)


def accuracy_ten(y_pred, y):
    y_pred = (y_pred *
              torch.arange(1, 11).unsqueeze(0).float()).mean(dim=1).round()
    y = (y * torch.arange(1, 11).unsqueeze(0).float()).mean(dim=1).round()
    return (y_pred == y).sum() / len(y)
