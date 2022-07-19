"""
Author: David Rozenberszki (david.rozenberszki@tum.de)
Date: Jan 07, 2022
"""
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import multiprocessing
from joblib import Parallel, delayed

from models.projection_models import AttributeFittingModel


class ContrastiveLanguageLoss(nn.Module):

    def __init__(self, config, num_labels, temperature=0.07, base_temperature=0.07, reduction='mean', feature_dim=512):

        super(ContrastiveLanguageLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_label = config.ignore_label
        self.config = config
        self.eps = 10e-5
        self.num_labels = num_labels
        self.clip_candidates = np.arange(num_labels)

        self.register_buffer('confusion_hist', torch.zeros((num_labels, num_labels)).long())
        if config.num_negative_samples > -1:
            self.num_negative_samples = config.num_negative_samples
        else:
            self.num_negative_samples = num_labels

        self.neg_thresh = config.contrast_neg_thresh
        self.pos_thresh = config.contrast_pos_thresh
        self.neg_weight = config.contrast_neg_weight

        # For multiprocessing
        self.num_cores = multiprocessing.cpu_count()

        self.reduction = reduction

        # The projection model that does the latent augmentation
        self.augment_categories = torch.empty(0)  # here we will store the category ids to be sampled for augmentation
        self.attributes = np.array(['A red ', 'A green ', 'A blue ', 'A yellow ', 'A dark ', 'A bright ', 'A big ', 'A small '])  # these are the allowed attributes
        self.augment_probability = config.instance_augmentation_color_aug_prob   # with this probability we will pick every instance to be augmented
        self.projection_model = AttributeFittingModel(feature_dim, feature_dim, self.attributes.shape[0])  # the pretrained model with the learned rotations as attributes

        model_path = config.scannet_path + '/' + config.projection_model_path
        if os.path.isfile(model_path):
            self.projection_model.load_state_dict(torch.load(model_path))
            print('Loaded weights for projection model')

        self.projection_model.eval()

    def latent_augmentation(self, features, labels):

            if random.random() < self.augment_probability:
                current_aug = np.random.randint(0, high=self.attributes.shape[0])
                projected_feats = self.projection_model(features)[:, current_aug]
                labels[:, 1] = current_aug
                return projected_feats, labels, current_aug + 1  # +1 bc the first index is the raw category
            else:
                return features, labels, 0



    def feat_dist(self, A, B, target):
        # takes two feature vectors and compute the point_wise distances
        # A = (n_points, feat_dim)
        # B = (n_points, num_samples, feat_dim)
        # Add zero loss to ignore labels

        if self.config.representation_distance_type == 'l2':  # L2 dist
            if B.shape[1] > 0:
                D2 = (A.unsqueeze(1) - B).pow(2).sum(dim=-1)
                loss = torch.sqrt(D2 + 1e-7).mean(1)
            else:
                loss = torch.zeros((A.size()))
        elif self.config.representation_distance_type == 'l1':  # L1 dist
            loss = (A.unsqueeze(1) - B).sum(dim=-1).mean(1)
        elif self.config.representation_distance_type == 'cos':  # Cos/Dot product dist
            # Multiply for batches as points
            loss = 1 - torch.bmm(F.normalize(A, p=2, dim=1).unsqueeze(1),
                                 F.normalize(B, p=2, dim=2).transpose(1, 2)).mean(-1).squeeze()  # (n_points, 1, feat_dim) @ (n_points, feat_dim, num_samples) = (n_points, 1, num_samples)
        else:
            return None  # Will throw error as it is not allowed

        loss[target == self.ignore_label] = 0.
        return loss

    def forward(self, features: torch.Tensor, labels: torch.Tensor, anchor_feats, preds=None):

        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [n_points, feat_dim]')

        # Push params to device after init
        if features.device != self.augment_categories.device:
            self.augment_categories = self.augment_categories.to(features.device)
        if features.device != next(self.projection_model.parameters()).device:
            self.projection_model = self.projection_model.to(features.device)

        point_num = features.shape[0]
        feat_dim = features.shape[1]

        # Init container for contrast tensors
        device = features.device
        pos_samples = torch.cuda.FloatTensor(point_num, 1, feat_dim).fill_(0)  # we have only 1 positive
        neg_samples = torch.cuda.FloatTensor(point_num, self.num_negative_samples, feat_dim).fill_(0)  # but can sample multiple negatives

        # Find feature samples for all labels
        if labels.dim() == 1:

            # Use only anchors without attributes
            if anchor_feats.dim() == 3:
                anchor_feats = anchor_feats[:, 0, :].squeeze()

            # Iterate over unique cats in feature tensor and remove ignore
            unique_targets = labels.unique()
            unique_targets = unique_targets[unique_targets != self.ignore_label]
            unique_targets_np = unique_targets.cpu().numpy()

            def target_samples(ut):

                # Find inds where we want to have contrasts
                ut_inds = labels == ut  # type: torch.Tensor
                ut_point_num = ut_inds.sum().item()

                # Pick pos samples
                pos_samples[ut_inds] = anchor_feats[ut].view(1, 1, feat_dim).expand(ut_point_num, 1, feat_dim)

                # Sample from all inds as ones with pos or self are given 0. prob
                if self.config.clip_uniform_sampling:
                    neg_candidates = self.clip_candidates[self.clip_candidates != ut.item()]
                else:
                    neg_candidates = unique_targets_np[unique_targets_np != ut.item()]

                sampled_neg_inds = torch.from_numpy(np.random.choice(neg_candidates, (ut_point_num, self.num_negative_samples))).to(device)
                neg_samples[ut_inds] = anchor_feats[sampled_neg_inds.view(-1)].view(ut_point_num, self.num_negative_samples, feat_dim)

            _ = Parallel(n_jobs=self.num_cores, backend="threading")(map(delayed(target_samples), unique_targets))

        else:  # In this case every target has a category and an attribute

            # Iterate over unique cats in feature tensor and remove ignore
            unique_targets = labels.unique(dim=0)
            unique_targets = unique_targets[unique_targets[:, 0] != self.ignore_label]
            unique_targets_np = unique_targets.cpu().numpy()

            def target_samples(ut):

                # Find inds where we want to have contrasts
                ut_inds = (labels[:, 0] == ut[0]) * (labels[:, 1] == ut[1])  # both the attribute and category has to match
                ut_point_num = ut_inds.sum().item()

                if self.config.instance_augmentation == 'latent':  # we do the encoded augmentation here
                    if ut[0] in self.augment_categories:
                        aug_feats, aug_labels, attribute_id = self.latent_augmentation(features[ut_inds], labels[ut_inds])
                        features[ut_inds, :] = aug_feats.to(features.device)
                        labels[ut_inds, :] = aug_labels.to(labels.device)
                        ut[1] = attribute_id

                # Pick pos samples
                pos_samples[ut_inds] = anchor_feats[ut[0], ut[1], :].view(1, 1, feat_dim).expand(ut_point_num, 1, feat_dim)

                # Sample from all inds as ones with pos or self are given 0. prob
                if self.config.clip_uniform_sampling:
                    neg_candidates = self.clip_candidates[self.clip_candidates != ut[0].item()]
                else:
                    neg_candidates = unique_targets_np[unique_targets_np != ut[0].item()]
                sampled_neg_inds = torch.from_numpy(np.random.choice(neg_candidates, (ut_point_num, self.num_negative_samples))).to(device)
                neg_samples[ut_inds] = anchor_feats[sampled_neg_inds.view(-1), 0, :].view(ut_point_num, self.num_negative_samples, feat_dim)  # here we sample only candidates without attributes

            _ = Parallel(n_jobs=self.num_cores, backend="threading")(map(delayed(target_samples), unique_targets))

            # Keep only categories after this and no attributes
            labels = labels[:, 0]

        # Use relu where error under threshold and only take losses for non ignored points
        pos_loss = F.relu(self.feat_dist(features, pos_samples, labels) - self.pos_thresh)
        neg_loss = F.relu(self.neg_thresh - self.feat_dist(features, neg_samples, labels))

        # Return weighted means
        if self.reduction == 'mean':
            loss = pos_loss.mean() + neg_loss.mean() * self.neg_weight
        else:
            loss = pos_loss + neg_loss * self.neg_weight

        return loss, pos_loss, neg_loss


class ContrastiveLanguageCELoss(ContrastiveLanguageLoss):

    def __init__(self, config, num_labels, temperature=0.07, base_temperature=0.07, reduction='mean'):

        super(ContrastiveLanguageCELoss, self).__init__(config, num_labels, temperature, base_temperature, reduction)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label, reduction=reduction)


    def feat_dist(self, A, B, target):

        if self.config.representation_distance_type == 'l2':
            if B.shape[1] > 0:
                D2 = (A.unsqueeze(1) - B).pow(2).sum(dim=-1)
                output = torch.sqrt(D2 + 1e-7)
            else:
                output = torch.zeros((A.size()))
        # Cos/Dot product dist
        elif self.config.representation_distance_type == 'cos':
            # Multiply for batches as points
            An = F.normalize(A, p=2, dim=1)
            output = torch.bmm(An.unsqueeze(1), B.transpose(1, 2)).squeeze()   # (n_points, 1, feat_dim) @ (n_points, feat_dim, num_samples) = (n_points, 1, num_samples)
        else:
            return None  # Will throw error as it is not allowed

        return output

    def forward(self, features: torch.Tensor, labels: torch.Tensor, anchor_feats, preds=None):

        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [n_points, feat_dim]')

        point_num = features.shape[0]
        feat_dim = features.shape[1]

        # Calculate feature distance with all anchors
        br_anchors = F.normalize(anchor_feats, p=2, dim=1).unsqueeze(0).expand(point_num, self.num_labels, feat_dim)
        dist_outputs = self.feat_dist(features, br_anchors, labels)

        loss = self.criterion(dist_outputs, labels)

        return loss, torch.zeros(1), loss