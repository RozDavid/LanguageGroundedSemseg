"""
Author: David Rozenberszki (david.rozenberszki@tum.de)
Date: Jan 29, 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing

from lib.losses.utils import sample_categories_for_balancing


class SupervisedSimSiam(nn.Module):

    def __init__(self, config, anchor_features, dataset):
        super(SupervisedSimSiam, self).__init__()

        # general global vars
        self.ignore_label = config.ignore_label
        self.config = config
        self.eps = 10e-5

        # For multiprocessing
        self.num_cores = multiprocessing.cpu_count()

        # Probably CLIP features to drive the representations
        self.anchor_features = anchor_features

        # Save dataset where it runs
        self.dataset = dataset

    def cosine_loss(self, A, B):
        An = F.normalize(A, p=2, dim=1)
        Bn = F.normalize(B, p=2, dim=1)
        return 1 - (An * Bn).sum(1)


    def forward(self, p1, p2, z1, z2, corrs1, corrs2, labels1, labels2):

        # Push to correct device if not already there
        device = p1.device
        if self.anchor_features.device != device:
            self.anchor_features = self.anchor_features.to(device)

        valid1 = labels1 != self.ignore_label
        valid2 = labels2 != self.ignore_label

        simsiam_loss1 = self.cosine_loss(p1[valid1], z2[corrs1][valid1])
        simsiam_loss2 = self.cosine_loss(p2[valid2], z1[corrs2][valid2])

        target_features1 = self.anchor_features[labels1[valid1]]
        target_features2 = self.anchor_features[labels2[valid2]]

        anchor_loss1 = self.cosine_loss(p1[valid1], target_features1)
        anchor_loss2 = self.cosine_loss(p2[valid2], target_features2)

        loss1 = (anchor_loss1) / 4.
        loss2 = (anchor_loss2) / 4.

        loss1, split_losses1, split_items1 = sample_categories_for_balancing(loss1, self.config, self.dataset, targets=labels1[valid1])
        loss2, split_losses2, split_items2 = sample_categories_for_balancing(loss2, self.config, self.dataset, targets=labels2[valid2])

        return loss1 + loss2, split_losses1, split_losses2, split_items1, split_items2


class PointSimSiamLoss(nn.Module):

    def __init__(self, config):
        super(PointSimSiamLoss, self).__init__()

        self.config = config
        self.eps = 10e-5

        # For multiprocessing
        self.num_cores = multiprocessing.cpu_count()

    def cosine_loss(self, A, B):
        An = F.normalize(A, p=2, dim=1)
        Bn = F.normalize(B, p=2, dim=1)
        return 1 - (An * Bn).sum(1)


    def forward(self, z1, z2, corrs1, corrs2):

        simsiam_loss = self.cosine_loss(z1, z2[corrs1])

        return simsiam_loss.mean()
