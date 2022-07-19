import torch
import torch.nn as nn
from lib.losses.FocalLoss import FocalLoss

def loss_by_name(loss_name, ignore_index=0, alpha=0.5, gamma=2.0, reduction='mean', weight=None):
    if loss_name == 'focal':
        return FocalLoss(alpha, gamma, reduction=reduction, ignore_index=ignore_index)
    elif loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    else:
        return None