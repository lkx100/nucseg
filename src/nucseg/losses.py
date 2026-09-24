"""Supervised loss and the NSL adversarial neighbour loss."""
import torch
import torch.nn.functional as F


def bce_dice(logits, target):
    """BCE plus soft Dice (per image, then averaged), equal weights."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    dice = (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + (1 - dice).mean()


def adversarial_neighbour(x, clean_loss, eps, scaler):
    """NSL adversarial neighbour of `x` (which must require grad): one signed-gradient step of size `eps`
    in [0, 1] pixel units, L-infinity norm. The AMP loss scale doesn't change the sign, so it doesn't change eps.
    The result is detached, so no second-order terms, as in NSL."""
    (g,) = torch.autograd.grad(scaler.scale(clean_loss), x, retain_graph=True)
    return (x + eps * torch.nan_to_num(g).sign()).clamp(0, 1).detach()
