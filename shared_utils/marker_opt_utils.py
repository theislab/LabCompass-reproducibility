import torch

#########################################################################################################
## LOSS FUNCTIONS
def l2_loss(
    pred,
    target,
    feature_mask=None,
):
    """
    pred: (N, M, D)
    target: (1, D)
    feature_mask: (D,)
    """
    # take mean when necessary
    pred = pred.mean(-2)
    err = pred - target

    if feature_mask is not None:
        err = err[..., feature_mask]

    val = torch.sum(err**2, dim=-1)
    return val


def l1_loss(
    pred,
    target,
    feature_mask=None,
):
    """
    pred: (N, M, D)
    target: (1, D)
    feature_mask: (D,)
    """
    # take mean when necessary
    pred = pred.mean(-2)
    err = pred - target

    if feature_mask is not None:
        err = err[..., feature_mask]

    val = torch.sum(torch.abs(err), dim=-1)
    return val
