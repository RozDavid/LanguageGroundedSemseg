"""
Author: David Rozenberszki (david.rozenberszki@tum.de)
Date: Jan 07, 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import multiprocessing
from joblib import Parallel, delayed


class PointSupConLoss(nn.Module):

    def __init__(self, config, num_labels, temperature=0.07, base_temperature=0.07, reduction='mean'):

        super(PointSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_label = config.ignore_label
        self.config = config
        self.eps = 10e-5
        self.num_labels = num_labels


        self.register_buffer('confusion_hist', torch.zeros((num_labels, num_labels)).long())
        self.num_pos_samples = config.num_pos_samples
        self.num_negative_samples = config.num_negative_samples

        self.neg_thresh = config.contrast_neg_thresh
        self.pos_thresh = config.contrast_pos_thresh
        self.neg_weight = config.contrast_neg_weight

        # For multiprocessing
        self.num_cores = multiprocessing.cpu_count()

        self.reduction = reduction

    def update_confusion_hist(self, new_confusion_hist):
        self.confusion_hist = new_confusion_hist + 1  # +1 to avoid summing up to 0 in the end

    def feat_dist(self, A, B, target):
        # takes two feature vectors and compute the point_wise distances
        # A = (n_points, feat_dim)
        # B = (n_points, num_samples, feat_dim)
        # Add zero loss to ignore labels

        # L2 dist
        distance = None
        if self.config.representation_distance_type == 'l2':
            if B.shape[1] > 0:
                D2 = (A.unsqueeze(1) - B).pow(2)
                D2 = torch.sum(D2, dim=-1)
                distance = torch.sqrt(D2 + 1e-7).mean(1)
            else:
                distance = torch.zeros((A.size()))
        # Cos/Dot product dist
        elif self.config.representation_distance_type == 'cos':
            # Multiply for batches as points
            An = F.normalize(A, p=2, dim=1)
            Bn = F.normalize(B, p=2, dim=2)
            Dcos = torch.bmm(An.unsqueeze(1), Bn.transpose(1, 2))  # (n_points, 1, feat_dim) @ (n_points, feat_dim, num_samples) = (n_points, 1, num_samples)
            Dcos = Dcos.mean(-1).squeeze()  # Average out over samples and remove unnecessary dims
            distance = 1 - Dcos  # Return the -1 * version as larger the more similar
        else:
            return None  # Will throw error as it is not allowed

        distance[target == self.ignore_label] = 0.
        return distance


    def forward(self, features: torch.Tensor, labels: torch.Tensor, anchor_feats=None, preds: torch.Tensor = None):

        device = features.device
        comp_feats = features.clone().detach()

        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [n_points, feat_dim]')

        point_num = features.shape[0]
        feat_dim = features.shape[1]
        labels = labels.contiguous()
        np_labels = labels.cpu().numpy()

        # Iterate over unique cats in feature tensor and remove ignore
        unique_targets = labels.unique()
        unique_targets = unique_targets[unique_targets != self.ignore_label]

        # Calculate valid inds for batch
        false_freq_mask = torch.zeros(self.num_labels).bool().to(device)
        false_freq_mask[unique_targets] = True

        # Init container for contrast tensors
        pos_samples = torch.zeros((point_num, self.num_pos_samples, feat_dim)).to(device)
        neg_samples = torch.zeros((point_num, self.num_negative_samples, feat_dim)).to(device)

        # Take into account only correct preds
        if preds is not None:
            correct_inds = (labels == preds)

        # Find feature samples for all labels
        #for ut in unique_targets:
        def target_smaples(ut):

            # Find inds where we want to have contrasts
            ut_inds = labels == ut  # type: torch.Tensor
            ut_point_num = ut_inds.sum().item()
            ut_index_values = np.arange(point_num)[ut_inds.cpu().numpy()]

            # #Filter for correct positives
            # if preds is None:
            #     ut_index_values = np.arange(point_num)[ut_inds.cpu().numpy()]
            # else:
            #     inds = torch.logical_and(correct_inds, ut_inds).cpu().numpy()
            #     ut_index_values = np.arange(point_num)[inds]

            # Pick pos samples
            pos_inds = torch.from_numpy(np.random.choice(ut_index_values, (ut_point_num, self.num_pos_samples))).to(device)
            pos_samples[ut_inds] = comp_feats[pos_inds.view(-1)].view(ut_point_num, self.num_pos_samples, feat_dim)

            # Calculate probs for sampling negative cats and only with ones present in scene, while masking out self
            ut_mask = false_freq_mask.clone()
            ut_mask[ut] = False
            false_freqs_prob = self.confusion_hist[ut].float() * ut_mask.float()
            false_freqs_prob /= false_freqs_prob.sum() + self.eps

            # map label to sample prob
            neg_probs = false_freqs_prob[labels]
            neg_probs[labels == self.ignore_label] = 0.

            # Take into account only correct preds for falses
            if preds is not None:
                neg_probs[~correct_inds] = 0.
            neg_probs = (neg_probs / (neg_probs.sum() + self.eps)).cpu().numpy()

            # Sample from all inds as ones with pos or self are given 0. prob
            sampled_neg_inds = torch.from_numpy(np.random.choice(np.arange(point_num), (ut_point_num, self.num_negative_samples), p=neg_probs)).to(device)
            neg_samples[ut_inds] = comp_feats[sampled_neg_inds.view(-1)].view(ut_point_num, self.num_negative_samples, feat_dim)

        _ = Parallel(n_jobs=self.num_cores, backend="threading")(map(delayed(target_smaples), unique_targets))

        # Use relu where error under threshold and only take losses for non ignored points
        pos_loss = F.relu(self.feat_dist(features, pos_samples, labels) - self.pos_thresh)
        neg_loss = F.relu(self.neg_thresh - self.feat_dist(features, neg_samples, labels))

        # Return weighted means
        if self.reduction == 'mean':
            loss = pos_loss.mean() + neg_loss.mean() * self.neg_weight
        else:
            loss = pos_loss + neg_loss * self.neg_weight

        return loss, pos_loss, neg_loss
