import torch
import math

INV_SQRT_TWO = 1 / math.sqrt(2)
INV_SQRT_TWOPI = 1 / math.sqrt(2 * math.pi)


def big_phi(x):
    return 0.5 * (1 + (x * INV_SQRT_TWO).erf())


def trunc_cdf_ava(x, mu, sigma, a=0.5, b=10.5):
    def normalize(val):
        return (val - mu) / (sigma + 1e-4)

    big_phi_x = big_phi(normalize(x))
    big_phi_a = big_phi(normalize(torch.tensor(a)))
    big_phi_b = big_phi(normalize(torch.tensor(b)))
    cdf = (big_phi_x - big_phi_a) / (big_phi_b - big_phi_a + 1e-4)
    if torch.sum(torch.isnan(cdf)):
        print(cdf)
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
        loss += 0.5 * y1 * (y1 / ys + 1).log()
        loss += 0.5 * y2 * (y2 / ys + 1).log()
        if torch.sum(torch.isnan(loss)):
            print(loss)
    return loss.mean()


def trunc_cjs_loss_R(y_true, y_pred):
    return cjs_loss(y_true, y_pred)


class TruncCJSLossD:
    def __init__(self, L1_D):
        self.L1_D = L1_D

    def __call__(self, y_true, y_pred, mask):
        loss = -cjs_loss(y_true, y_pred)
        loss += self.L1_D * mask.abs().mean()
        return loss


def cjs_loss_10(y_true, y_pred):
    loss = 0
    y1 = y_true.cumsum(dim=1) + 1e-5
    y2 = y_pred.cumsum(dim=1) + 1e-5
    ys = 0.5 * (y1 + y2)
    loss += 0.5 * y1 * (y1 / ys).log()
    loss += 0.5 * y2 * (y2 / ys).log()
    return loss.sum(dim=1).mean()


class CJSLoss10R:
    def __init__(self, L1_R):
        self.L1_R = L1_R

    def __call__(self, y_true, y_pred, ill_features):
        loss = cjs_loss_10(y_true, y_pred)
        # l1_term = ill_features.abs().mean()
        # loss += 0.6 * l1_term
        return loss


class CJSLoss10D:
    def __init__(self, L1_D):
        self.L1_D = L1_D

    def __call__(self, y_true, y_pred, mask):
        loss = -cjs_loss_10(y_true, y_pred)
        loss += -4 * loss.detach().item() * mask.abs().mean()
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
        return loss
