import torch
import math

INV_SQRT_TWO = 1 / math.sqrt(2)
INV_SQRT_TWOPI = 1 / math.sqrt(2 * math.pi)


def l1_penalty(params):
    l1_norm = sum(p.abs().sum() for p in params)
    return l1_norm


def big_phi(x):
    return 0.5 * (1 + (x * INV_SQRT_TWO).erf())


def trunc_cdf_ava(x, mu, sigma, a=0.5, b=10.5):
    def normalize(val):
        return (val - mu) / sigma

    big_phi_x = big_phi(normalize(x))
    big_phi_a = big_phi(normalize(torch.tensor(a)))
    big_phi_b = big_phi(normalize(torch.tensor(b)))
    cdf = (big_phi_x - big_phi_a) / (big_phi_b - big_phi_a + 1e-4)
    if torch.sum(torch.isnan(cdf)):
        print(x, mu, sigma, a, b)
    return cdf


def cjs_loss(y_true, y_pred):
    loss = 0
    y1s = [
        trunc_cdf_ava(i + 1.5, y_true[:, 0], y_true[:, 1]) for i in range(10)
    ]
    y2s = [
        trunc_cdf_ava(i + 1.5, y_pred[:, 0], y_pred[:, 1]) for i in range(10)
    ]
    for i in range(10):
        y1 = y1s[i]
        y2 = y2s[i]
        ys = 0.5 * (y1 + y2)
        loss += 0.5 * y1 * (y1 + 1e-4 / ys + 1e-4).log()
        loss += 0.5 * y2 * (y2 + 1e-4 / ys + 1e-4).log()
        if torch.sum(torch.isnan(loss)):
            print(min(y1), min(y2), min(ys))
    return loss.mean()


def trunc_cjs_loss_R(y_true, y_pred):
    return cjs_loss(y_true, y_pred)


class TruncCJSLossD:
    def __init__(self, L1_D):
        self.L1_D = L1_D

    def __call__(self, y_true, y_pred, mask):
        loss = -cjs_loss(y_true, y_pred)
        loss += self.L1_D * mask.abs().sum()
        return loss


####


def toy_loss_R(y_true, y_pred):
    return ((y_true[:, 0] - y_pred[:, 0])**2).mean()**0.5


# remember to include the L1 penelty
class ToyLossD:
    def __init__(self, L1_D):
        self.L1_D = L1_D

    def __call__(self, y_true, y_pred, mask):
        loss = -((y_true[:, 0] - y_pred[:, 0])**2).mean()**0.5
        loss += self.L1_D * mask.abs().sum()
        print(loss)
        return loss
