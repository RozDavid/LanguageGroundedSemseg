import io
import json
import logging
import os
import errno
import pickle
import time

import numpy as np
import sklearn.metrics
import torch
from torch import nn

from lib.pc_utils import colorize_pointcloud, save_point_cloud
from lib.losses.FocalLoss import FocalLoss

def load_state_with_same_shape(model, weights, prefix=''):

    model_state = model.state_dict()
    if list(weights.keys())[0].startswith('module.'):
        logging.info("Loading multigpu weights with module. prefix...")
        weights = {k.partition('module.')[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('model.'):
        logging.info("Loading Pytorch-Lightning weights from state")
        weights = {k.partition('model.')[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('encoder.'):
        logging.info("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition('encoder.')[2]: weights[k] for k in weights.keys()}

    # print(weights.items())
    # print("===================")
    # print("===================")
    # print("===================")
    # print("===================")
    # print("===================")
    # print(model_state)

    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))

    return filtered_weights


def checkpoint(model, optimizer, epoch, iteration, config, best_val=None, best_val_iter=None, postfix=None):
    mkdir_p(config.log_dir)
    if config.overwrite_weights:
        if postfix is not None:
            filename = f"checkpoint_{config.wrapper_type}{config.model}{postfix}.pth"
        else:
            filename = f"checkpoint_{config.wrapper_type}{config.model}.pth"
    else:
        filename = f"checkpoint_{config.wrapper_type}{config.model}_iter_{iteration}.pth"
    checkpoint_file = config.log_dir + '/' + filename
    state = {
        'iteration': iteration,
        'epoch': epoch,
        'arch': config.model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if best_val is not None:
        state['best_val'] = best_val
        state['best_val_iter'] = best_val_iter
    json.dump(vars(config), open(config.log_dir + '/config.json', 'w'), indent=4)
    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")
    # Delete symlink if it exists
    if os.path.exists(f'{config.log_dir}/weights.pth'):
        os.remove(f'{config.log_dir}/weights.pth')
    # Create symlink
    os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')


def precision_at_one(pred, target, ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != ignore_label]
    correct = correct.view(-1)
    if correct.nelement():
        return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
    else:
        return 0.


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def fast_hist_torch(pred, label, n):
    k = (label >= 0) & (label < n)
    return torch.bincount(n * label[k].int() + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def per_class_iu_torch(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))


def loss_by_name(loss_name, ignore_index=0, alpha=0.5, gamma=2.0, reduction='mean', weight=None):
    if loss_name == 'focal':
        return FocalLoss(alpha, gamma, reduction=reduction, ignore_index=ignore_index)
    elif loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    else:
        return None

def save_obj(output_path, obj):
    with open(output_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(intput_path ):
    with open(intput_path, 'rb') as f:
        return pickle.load(f)

class WithTimer(object):
    """Timer for with statement."""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        out_str = 'Elapsed: %s' % (time.time() - self.tstart)
        if self.name:
            logging.info('[{self.name}]')
        logging.info(out_str)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.averate_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True, with_call=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        if with_call or self.calls == 0:
            self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ExpTimer(Timer):
    """ Exponential Moving Average Timer """

    def __init__(self, alpha=0.5):
        super(ExpTimer, self).__init__()
        self.alpha = alpha

    def toc(self):
        self.diff = time.time() - self.start_time
        self.average_time = self.alpha * self.diff + \
                            (1 - self.alpha) * self.average_time
        return self.average_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 10e-5)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def debug_on():
    import sys
    import pdb
    import functools
    import traceback

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper

    return decorator


def get_prediction(dataset, output, target):
    return output.max(1)[1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
    return torch.device('cuda' if is_cuda else 'cpu')


class HashTimeBatch(object):

    def __init__(self, prime=5279):
        self.prime = prime

    def __call__(self, time, batch):
        return self.hash(time, batch)

    def hash(self, time, batch):
        return self.prime * batch + time

    def dehash(self, key):
        time = key % self.prime
        batch = key / self.prime
        return time, batch


def save_rotation_pred(iteration, pred, dataset, save_pred_dir):
    """Save prediction results in original pointcloud scale."""
    decode_label_map = {}
    for k, v in dataset.label_map.items():
        decode_label_map[v] = k
    pred = np.array([decode_label_map[x] for x in pred], dtype=np.int)
    out_rotation_txt = dataset.get_output_id(iteration) + '.txt'
    out_rotation_path = save_pred_dir + '/' + out_rotation_txt
    np.savetxt(out_rotation_path, pred, fmt='%i')


def save_predictions(coords, upsampled_pred, transformation, dataset, config, iteration,
                     save_pred_dir):
    """Save prediction results in original pointcloud scale."""
    from lib.dataset import OnlineVoxelizationDatasetBase
    if dataset.IS_ONLINE_VOXELIZATION:
        assert transformation is not None, 'Need transformation matrix.'
    iter_size = coords[:, -1].max() + 1  # Normally batch_size, may be smaller at the end.
    if dataset.IS_TEMPORAL:  # Iterate over temporal dilation.
        iter_size *= config.temporal_numseq
    for i in range(iter_size):
        # Get current pointcloud filtering mask.
        if dataset.IS_TEMPORAL:
            j = i % config.temporal_numseq
            i = i // config.temporal_numseq
        batch_mask = coords[:, 0].numpy() == i
        if dataset.IS_TEMPORAL:
            batch_mask = np.logical_and(batch_mask, coords[:, -1].numpy() == j)
        # Calculate original coordinates.
        coords_original = coords[:, 1:].numpy()[batch_mask] + 0.5
        if dataset.IS_ONLINE_VOXELIZATION:
            # Undo voxelizer transformation.
            curr_transformation = transformation[i, :16].numpy().reshape(4, 4)
            xyz = np.hstack((coords_original, np.ones((batch_mask.sum(), 1))))
            orig_coords = (np.linalg.inv(curr_transformation) @ xyz.T).T
        else:
            orig_coords = coords_original
        orig_pred = upsampled_pred[batch_mask]
        # Undo ignore label masking to fit original dataset label.
        if dataset.IGNORE_LABELS:
            if isinstance(dataset, OnlineVoxelizationDatasetBase):
                label2masked = dataset.label2masked
                maskedmax = label2masked[label2masked < 2000].max() + 1
                masked2label = [label2masked.tolist().index(i) for i in range(maskedmax)]
                orig_pred = np.take(masked2label, orig_pred)
            else:
                decode_label_map = {}
                for k, v in dataset.label_map.items():
                    decode_label_map[v] = k
                orig_pred = np.array([decode_label_map[x] for x in orig_pred], dtype=np.int)
        # Determine full path of the destination.
        full_pred = np.hstack((orig_coords[:, :3], np.expand_dims(orig_pred, 1)))
        filename = 'pred_%04d_%02d.npy' % (iteration, i)
        if dataset.IS_TEMPORAL:
            filename = 'pred_%04d_%02d_%02d.npy' % (iteration, i, j)
        # Save final prediction as npy format.
        np.save(os.path.join(save_pred_dir, filename), full_pred)


def visualize_results(coords, colors, target,
                      prediction, config,
                      iteration, num_labels,
                      train_iteration=None,
                      valid_labels=None,
                      save_npy=False,
                      scene_name='',
                      refinement_pred=None,
                      refinement_target=None,
                      allow_0=False,
                      output_features=None):
    if train_iteration:
        base_file_name = '_'.join([config.dataset, config.model, 'train_{}'.format(train_iteration)])
    else:
        base_file_name = '_'.join([config.dataset, config.model, 'test'])

    # Create directory to save visualization results.
    os.makedirs(config.visualize_path, exist_ok=True)

    if refinement_pred is not None and refinement_pred.dim() == 1:
        refinement_pred = refinement_pred[:, None]
        refinement_target = refinement_target[:, None]

    # Get filter for valid predictions in the first batch.
    target_batch = (coords[:, 0] == 0).cpu()
    input_xyz = coords[:, 1:]
    target_valid = torch.ne(target, config.ignore_label)
    batch_ids = torch.logical_and(target_batch, target_valid)
    target_nonpred = torch.logical_and(target_batch, ~target_valid)  # type: torch.Tensor
    ptc_nonpred = np.hstack(
        (input_xyz[target_nonpred].cpu().numpy(), np.zeros((torch.sum(target_nonpred).item(), 3))))  # type: np.ndarray
    ptc_nonpred_np = np.hstack(
        (input_xyz[target_nonpred].cpu().numpy(), np.zeros((torch.sum(target_nonpred).item(), 1))))  # type: np.ndarray

    scaled_input_cords = coords[:, 1:].int()
    input_target_batch = coords[:, 0] == 0  # type: torch.Tensor
    scaled_input_feats = (colors + 0.5) * 255.
    scaled_input_feats = scaled_input_feats.int()  # type: torch.Tensor

    # Predcited label visualization in RGB.
    input_xyz_np = input_xyz[batch_ids].cpu().numpy()
    xyzlabel = colorize_pointcloud(input_xyz_np, prediction[batch_ids.numpy()], num_labels)  # type: np.ndarray
    xyzlabel_np_pred = np.hstack((input_xyz_np, prediction[batch_ids.numpy()][:, None]))  # type: np.ndarray
    xyzlabel = np.vstack((xyzlabel, ptc_nonpred))  # type: np.ndarray
    xyzlabel_np_pred = np.vstack((xyzlabel_np_pred, ptc_nonpred_np))  # type: np.ndarray
    filename_pred = '_'.join([base_file_name, 'pred', '%04d.ply' % iteration])
    if refinement_pred is not None:
        refinement = refinement_pred.cpu().numpy()[batch_ids]
        refinement_nonpred = refinement_pred.cpu().numpy()[target_nonpred]
        refinement = np.vstack((refinement, refinement_nonpred))
        xyzlabel = np.hstack((xyzlabel, refinement))
    save_point_cloud(xyzlabel, os.path.join(config.visualize_path, filename_pred), with_refinement=True, verbose=False)

    # RGB input values visualization.
    xyzrgb = torch.hstack(
        (scaled_input_cords[input_target_batch], scaled_input_feats[:, :3][input_target_batch]))  # type: torch.Tensor
    filename = '_'.join([base_file_name, 'rgb', '%04d.ply' % iteration])
    save_point_cloud(xyzrgb.cpu().numpy(), os.path.join(config.visualize_path, filename), verbose=False)

    # Ground-truth visualization in RGB.
    xyzgt = colorize_pointcloud(input_xyz_np, target.numpy()[batch_ids], num_labels)  # type: np.ndarray
    xyzgt_np = np.hstack((input_xyz_np, np.expand_dims(target.numpy()[batch_ids], axis=1)))  # type: np.ndarray

    xyzgt = np.vstack((xyzgt, ptc_nonpred))  # type: np.ndarray
    xyzgt_np = np.vstack((xyzgt_np, ptc_nonpred_np))  # type: np.ndarray
    if refinement_pred is not None:
        refinement = refinement_target.cpu().numpy()[batch_ids]
        refinement_nonpred = refinement_target.cpu().numpy()[target_nonpred]
        refinement = np.vstack((refinement, refinement_nonpred))
        xyzgt = np.hstack((xyzgt, refinement))
    filename = '_'.join([base_file_name, 'gt', '%04d.ply' % iteration])
    save_point_cloud(xyzgt, os.path.join(config.visualize_path, filename), with_refinement=True, verbose=False)

    # Finally save confusion matrix
    valid_targets = xyzgt_np[:, -1] != 0 if not allow_0 else xyzgt_np[:, -1] == xyzgt_np[:, -1]
    confusion_matrix = sklearn.metrics.confusion_matrix(xyzgt_np[valid_targets, -1],
                                                        xyzlabel_np_pred[valid_targets, -1], labels=valid_labels)
    filename_conf = '_'.join([base_file_name, 'confusion', '%04d.pkl' % iteration])
    confusion_pkl = {'scene_name': scene_name, 'confusion_mat': confusion_matrix}
    save_obj(os.path.join(config.visualize_path, filename_conf), confusion_pkl)

    if save_npy:
        filename_pred_np = '_'.join([base_file_name, 'pred', '%04d.npy' % iteration])
        np.save(os.path.join(config.visualize_path, filename_pred_np), xyzlabel_np_pred)
        filename_np = '_'.join([base_file_name, 'gt', '%04d.npy' % iteration])
        np.save(os.path.join(config.visualize_path, filename_np), xyzgt_np)

    if refinement_pred is not None and refinement_target is not None:
        filename_pred_refinement_np = '_'.join([base_file_name, 'pred_refinement', '%04d.npy' % iteration])
        np.save(os.path.join(config.visualize_path, filename_pred_refinement_np), refinement_pred.cpu().numpy())
        filename_target_refinement_np = '_'.join([base_file_name, 'gt_refinement', '%04d.npy' % iteration])
        np.save(os.path.join(config.visualize_path, filename_target_refinement_np), refinement_target.cpu().numpy())

    if output_features is not None:
        filename_features_np = '_'.join([base_file_name, 'final_feats', scene_name[0]])
        np.save(os.path.join(config.visualize_path, filename_features_np), output_features.cpu().numpy())


def save_target_freqs(freqs_dict, target_sum_losses, features_dict, iteration, config):

    base_file_name = '_'.join([config.dataset, config.model, 'train_{}'.format(iteration), "target_frequencies.pkl"])
    losses_file_name = '_'.join([config.dataset, config.model, 'train_{}'.format(iteration), "mean_losses_by_targets.pkl"])
    features_file_name = '_'.join([config.dataset, config.model, 'train_{}'.format(iteration), "sampled_cat_features.pkl"])

    # Create directory to save visualization results.
    os.makedirs(config.visualize_path, exist_ok=True)

    # Normalize losses to mean
    if target_sum_losses is not None:
        for cat, l in target_sum_losses.items():
            target_sum_losses[cat] = (target_sum_losses[cat] / freqs_dict[cat]).cpu().numpy()
        full_losses_path = os.path.join(config.visualize_path, losses_file_name)
        with open(full_losses_path, 'wb') as f:
            pickle.dump(target_sum_losses, f, pickle.HIGHEST_PROTOCOL)

    if freqs_dict is not None:
        full_path = os.path.join(config.visualize_path, base_file_name)
        with open(full_path, 'wb') as f:
            pickle.dump(freqs_dict, f, pickle.HIGHEST_PROTOCOL)

    if features_dict is not None:
        full_features_path = os.path.join(config.visualize_path, features_file_name)
        with open(full_features_path, 'wb') as f:
            pickle.dump(features_dict, f, pickle.HIGHEST_PROTOCOL)


def save_feature_maps(feature_maps, config, scene_name, targets=None, coords=None):

    base_file_name = '_'.join([scene_name, "feature_maps.pkl"])
    # Create directory to save visualization results.
    os.makedirs(config.visualize_path, exist_ok=True)

    out_dict = {'feature_map': feature_maps}
    if targets is not None:
        out_dict = {'target': targets, **out_dict}

    if coords is not None:
        out_dict = {'coords': coords, **out_dict}

    base_file_name = os.path.join(config.visualize_path, base_file_name)
    with open(base_file_name, 'wb') as f:
        pickle.dump(out_dict, f, pickle.HIGHEST_PROTOCOL)

def save_mean_features(mean_features, iteration, config):

    base_file_name = '_'.join([config.dataset, config.model, 'train_{}'.format(iteration), "mean_features.pkl"])

    # Create directory to save visualization results.
    os.makedirs(config.visualize_path, exist_ok=True)

    full_path = os.path.join(config.visualize_path, base_file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(mean_features, f, pickle.HIGHEST_PROTOCOL)


def map_to_orig_labels(parent_labels, child_labels, parent_mapping, child_mapping,
                       supercat_labels=None, supercat_mapping=None):
    parent_mapper = lambda t: parent_mapping[t]
    child_mapper = lambda t: child_mapping[t]

    parent_labels.apply_(parent_mapper)
    child_labels.apply_(child_mapper)

    if supercat_labels is not None and supercat_mapping is not None:
        supercat_mapper = lambda t: supercat_mapping[t]
        supercat_labels.apply_(supercat_mapper)

        return supercat_labels, parent_labels, child_labels

    else:
        return parent_labels, child_labels


def orig2general(labels, mapping):
    mapper = lambda t: mapping[t]
    labels.apply_(mapper)

    return labels


def permute_pointcloud(input_coords, pointcloud, transformation, label_map,
                       voxel_output, voxel_pred):
    """Get permutation from pointcloud to input voxel coords."""

    def _hash_coords(coords, coords_min, coords_dim):
        return np.ravel_multi_index((coords - coords_min).T, coords_dim)

    # Validate input.
    input_batch_size = input_coords[:, -1].max().item()
    pointcloud_batch_size = pointcloud[:, -1].max().int().item()
    transformation_batch_size = transformation[:, -1].max().int().item()
    assert input_batch_size == pointcloud_batch_size == transformation_batch_size
    pointcloud_permutation, pointcloud_target = [], []

    # Process each batch.
    for i in range(input_batch_size + 1):
        # Filter batch from the data.
        input_coords_mask_b = input_coords[:, -1] == i
        input_coords_b = (input_coords[input_coords_mask_b])[:, :-1].numpy()
        pointcloud_b = pointcloud[pointcloud[:, -1] == i, :-1].numpy()
        transformation_b = transformation[i, :-1].reshape(4, 4).numpy()
        # Transform original pointcloud to voxel space.
        original_coords1 = np.hstack((pointcloud_b[:, :3], np.ones((pointcloud_b.shape[0], 1))))
        original_vcoords = np.floor(original_coords1 @ transformation_b.T)[:, :3].astype(int)
        # Hash input and voxel coordinates to flat coordinate.
        vcoords_all = np.vstack((input_coords_b, original_vcoords))
        vcoords_min = vcoords_all.min(0)
        vcoords_dims = vcoords_all.max(0) - vcoords_all.min(0) + 1
        input_coords_key = _hash_coords(input_coords_b, vcoords_min, vcoords_dims)
        original_vcoords_key = _hash_coords(original_vcoords, vcoords_min, vcoords_dims)
        # Query voxel predictions from original pointcloud.
        key_to_idx = dict(zip(input_coords_key, range(len(input_coords_key))))
        pointcloud_permutation.append(
            np.array([key_to_idx.get(i, -1) for i in original_vcoords_key]))
        pointcloud_target.append(pointcloud_b[:, -1].astype(int))
    pointcloud_permutation = np.concatenate(pointcloud_permutation)
    # Prepare pointcloud permutation array.
    pointcloud_permutation = torch.from_numpy(pointcloud_permutation)
    permutation_mask = pointcloud_permutation >= 0
    permutation_valid = pointcloud_permutation[permutation_mask]
    # Permute voxel output to pointcloud.
    pointcloud_output = torch.zeros(pointcloud.shape[0], voxel_output.shape[1]).to(voxel_output)
    pointcloud_output[permutation_mask] = voxel_output[permutation_valid]
    # Permute voxel prediction to pointcloud.
    # NOTE: Invalid points (points found in pointcloud but not in the voxel) are mapped to 0.
    pointcloud_pred = torch.ones(pointcloud.shape[0]).int().to(voxel_pred) * 0
    pointcloud_pred[permutation_mask] = voxel_pred[permutation_valid]
    # Map pointcloud target to respect dataset IGNORE_LABELS
    pointcloud_target = torch.from_numpy(
        np.array([label_map[i] for i in np.concatenate(pointcloud_target)])).int()
    return pointcloud_output, pointcloud_pred, pointcloud_target


def nanmean_t(torch_array):
    value = torch_array[~torch.isnan(torch_array)].mean().item()
    if np.isnan(value):
        return 0.
    else:
        return value


def print_info(iteration,
               max_iteration,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None,
               dataset_frequency_cats=None):
    debug_str = "{}/{}: ".format(iteration, max_iteration)

    acc = (hist.diagonal() / hist.sum(1) * 100)
    debug_str += "\tAVG Loss {loss:.3f}\t" \
                 "AVG Score {top1:.3f}\t" \
                 "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
        loss=losses.item(), top1=scores.item(), mIOU=np.nanmean(ious),
        mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))

    if dataset_frequency_cats is not None:
         debug_str += 'Head mIoU {head:.3f} \t Common mIoU {common:.3f} \tTail mIoU {tail:.3f} \n'.format(head=np.nanmean(ious[dataset_frequency_cats[:, 0]]),
                                                                                                         common=np.nanmean(ious[dataset_frequency_cats[:, 1]]),
                                                                                                         tail=np.nanmean(ious[dataset_frequency_cats[:, 2]]))

    if class_names is not None:
        debug_str += "\nClasses: " + ", ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ', '.join('{:.03f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAP: ' + ', '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
    debug_str += 'mAcc: ' + ', '.join('{:.03f}'.format(i) for i in acc) + '\n'

    logging.info(debug_str)