import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric

def embedding_loss(embedding, target, feature_clusters, criterion, config):
    # Or triplet loss with semi-hard negative sampling
    embed_target = feature_clusters.to(target.device)[target]
    loss = criterion(embedding, embed_target)
    return torch.mean(loss[target != config.ignore_label], dim=1)


def sample_categories_for_balancing(loss, config, dataset, targets, outputs=None):

    # This means only the valid inds are kept

    if loss.size(0) != targets.size(0):
        valid_mask = targets != config.ignore_label
        targets = targets[valid_mask]

    all_inds = torch.arange(loss.size(0)).to(loss.device)
    category_inds = torch.arange(dataset.NUM_LABELS).to(targets.device)

    # Skip if ignore label
    targets_valid = targets[targets != config.ignore_label]

    head_inds = category_inds[dataset.frequency_organized_cats[:, 0]]
    common_inds = category_inds[dataset.frequency_organized_cats[:, 1]]
    tail_inds = category_inds[dataset.frequency_organized_cats[:, 2]]

    head_sample_ratio = config.balanced_sample_head_ratio
    common_sample_ratio = config.balanced_sample_common_ratio
    point_loss_mask = np.zeros(loss.size(0)).astype(bool)

    loss_items = torch.zeros((loss.size(0), 3)).bool().to(loss.device)

    u_values, u_counts = targets_valid.unique(return_counts=True)
    for unique_target, unique_count in zip(u_values, u_counts):
        # Sample if head or common:
        if unique_target in head_inds:
            target_inds = all_inds[targets == unique_target]
            if head_sample_ratio > 0.:
                sampled_targets = np.random.choice(target_inds.cpu().numpy(),
                                                   round(head_sample_ratio * unique_count.item()),
                                                   replace=False)
            else:
                sampled_targets = all_inds[targets == unique_target].cpu().numpy()

            point_loss_mask[sampled_targets] = True
            loss_items[target_inds, 0] = True

        elif unique_target in common_inds:
            target_inds = all_inds[targets == unique_target]
            if common_sample_ratio > 0.:
                sampled_targets = np.random.choice(target_inds.cpu().numpy(),
                                                   round(common_sample_ratio * unique_count.item()),
                                                   replace=False)
            else:
                sampled_targets = all_inds[targets == unique_target].cpu().numpy()

            point_loss_mask[sampled_targets] = True
            loss_items[target_inds, 1] = True
        else:
            # Keep all samples for tail cats
            target_inds = all_inds[targets == unique_target].cpu().numpy()
            point_loss_mask[target_inds] = True
            loss_items[target_inds, 2] = True

    # Calculate split losses
    head_loss = loss[loss_items[:, 0]].detach()
    common_loss = loss[loss_items[:, 1]].detach()
    tail_loss = loss[loss_items[:, 2]].detach()

    # Finally, mask the loss
    loss = loss * torch.Tensor(point_loss_mask).to(loss.device)

    return loss.mean(), (head_loss, common_loss, tail_loss), loss_items[targets != config.ignore_label, :]


def feature_sim(output_feats, anchor_feats, config):

    # Take only non attributed features if multiple is available
    if anchor_feats.dim() == 3:
        anchor_feats = anchor_feats[:, 0, :].squeeze()

    if config.representation_distance_type == 'l2':
        # need this for memory constraint
        D2 = torch.zeros(output_feats.shape[0], anchor_feats.shape[0]).to(output_feats.device)  # dist for every point to the anchor category
        for i in range(anchor_feats.shape[0]):  # iterate over categories
            D2[:, i] = (output_feats - anchor_feats[i, :]).pow(2).sum(dim=-1)
        return -D2  # we need to return the inverse to find the most similar with argmax
    elif config.representation_distance_type == 'l1':
        # need this for memory constraint
        D1 = torch.zeros(output_feats.shape[0], anchor_feats.shape[0]).to(output_feats.device)  # dist for every point to the anchor category
        for i in range(anchor_feats.shape[0]):  # iterate over categories
            d1 = (output_feats - anchor_feats[i, :]).sum(dim=-1)
            D1[:, i] = d1
        return -D1  # we need to return the inverse to find the most similar with argmax
    else:  # cosine
        An = F.normalize(output_feats, p=2, dim=1)
        Bn = F.normalize(anchor_feats, p=2, dim=1).unsqueeze(0).expand(An.shape[0], anchor_feats.shape[0], anchor_feats.shape[1])
        Dcos = torch.bmm(An.unsqueeze(1), Bn.transpose(1, 2))
        return Dcos.squeeze()


class MetricAverageMeter(Metric):
    def __init__(self, ignore_index=-1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("value", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ignore_index = ignore_index

    def update(self, value, count: int = None):
        self.total += count
        self.value += torch.tensor(value * count).to(self.device)

    def compute(self):
        return self.value / self.total
