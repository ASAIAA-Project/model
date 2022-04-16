def l1_penalty(params, l1_lambda=0.001):
    l1_norm = sum(p.abs().sum() for p in params)
    return l1_lambda * l1_norm
