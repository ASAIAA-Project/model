def l1_penalty(params):
    l1_norm = sum(p.abs().sum() for p in params)
    return l1_norm


def toy_loss_R(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()**0.5


# remeber to include the L1 penelty
class ToyLossD:
    def __init__(self, L1_D):
        self.L1_D = L1_D

    def __call__(self, y_true, y_pred, mask):
        loss = -((y_true - y_pred)**2).mean()**0.5
        loss += self.L1_D * mask.abs().sum()
        return loss
