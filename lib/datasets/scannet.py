import glob
import logging
import os
import random
import sys
from pathlib import Path
import pickle

import numpy as np
from scipy import spatial, ndimage, misc
import torch

from lib.constants.dataset_sets import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.transforms import InstanceAugmentation
from lib.utils import read_txt, fast_hist, per_class_iu

from lib.constants.scannet_constants import *
from lib.datasets.preprocessing.utils import box_intersect

import MinkowskiEngine as ME

class ScannetVoxelizationDataset(VoxelizationDataset):

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_20
    CLASS_LABELS = CLASS_LABELS_20
    VALID_CLASS_IDS = VALID_CLASS_IDS_20

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    NUM_LABELS = max(SCANNET_COLOR_MAP_LONG.keys())  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
    IS_FULL_POINTCLOUD_EVAL = True

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'train.txt',
        DatasetPhase.Val: 'val.txt',
        DatasetPhase.TrainVal: 'trainval.txt',
        DatasetPhase.Test: 'test.txt'
    }

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 phase=DatasetPhase.Train):
        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        # Use cropped rooms for train/val
        data_root = config.scannet_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
        logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
        super().__init__(
            data_paths,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.ignore_label,
            return_transformation=config.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config)

        # Load category weights for weighted CE and Focal
        self.category_weights = torch.ones(self.NUM_LABELS)
        category_weights_path = config.scannet_path + '/' + config.category_weights
        if os.path.isfile(category_weights_path):
            with open(category_weights_path, "rb") as input_file:
                category_weights = pickle.load(input_file)
                print('Loaded category weights for CE {}'.format(category_weights_path))

            for cat_id, cat_value in category_weights.items():
                if cat_id > 0:
                    mapped_id = self.label_map[cat_id]
                    self.category_weights[mapped_id] = cat_value

        # Load instance sampling weights for instance based balancing
        self.instance_sampling_weights = np.ones(self.NUM_LABELS)
        instance_sampling_weights_path = config.scannet_path + '/' + config.instance_sampling_weights
        if os.path.isfile(instance_sampling_weights_path) and config.sample_tail_instances:
            with open(instance_sampling_weights_path, "rb") as input_file:
                instance_sampling_weights = pickle.load(input_file)
                print('Loaded instance sampling probability weights {}'.format(instance_sampling_weights_path))
            for cat_id, cat_value in instance_sampling_weights.items():
                if cat_id > 0:
                    mapped_id = self.label_map[cat_id]
                    self.instance_sampling_weights[mapped_id] = cat_value
        self.instance_sampling_weights /= self.instance_sampling_weights.sum()

        # Precompute a mapping from ids to categories
        self.id2cat_name = {}
        for id, cat_name in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            self.id2cat_name[id] = cat_name

        # Load the bounding boxes from all instances of the dataset
        bb_path = config.scannet_path + '/' + config.bounding_boxes_path
        if os.path.isfile(bb_path):
            with open(bb_path, 'rb') as f:
                self.bounding_boxes = pickle.load(f)

        # To use instance level augmentation like color or scale shift
        self.instance_augmentation_transform = InstanceAugmentation(config)
        self.aug_color_prob = config.instance_augmentation_color_aug_prob
        self.aug_scale_prob = config.instance_augmentation_scale_aug_prob

        # Calculate head-common-tail ids
        self.head_ids = []
        self.common_ids = []
        self.tail_ids = []
        self.frequency_organized_cats = torch.zeros(self.NUM_LABELS, 3).bool()
        for scannet_id, scannet_cat in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            if scannet_cat in HEAD_CATS_SCANNET_200:
                self.head_ids += [self.label_map[scannet_id]]
                self.frequency_organized_cats[self.label_map[scannet_id], 0] = True
            elif scannet_cat in COMMON_CATS_SCANNET_200:
                self.common_ids += [self.label_map[scannet_id]]
                self.frequency_organized_cats[self.label_map[scannet_id], 1] = True
            elif scannet_cat in TAIL_CATS_SCANNET_200:
                self.tail_ids += [self.label_map[scannet_id]]
                self.frequency_organized_cats[self.label_map[scannet_id], 2] = True

    def add_instances_to_cloud(self, coords, feats, labels, scene_name, transformations):

        if self.config.is_train:
            phase = 'train'
        else:
            phase = 'val'

        coords = coords.astype(int)
        voxel_scale, trans_rot = transformations

        instance_folder = self.config.scannet_path + f'/train/{phase}_instances/'
        num_instances = self.config.num_instances_to_add
        samples = np.random.choice(self.VALID_CLASS_IDS, num_instances, p=self.instance_sampling_weights)
        scene_bbs = self.bounding_boxes[scene_name]

        # Get scene dimensions
        scene_maxes = np.amax(coords, axis=0)
        scene_mins = np.amin(coords, axis=0)
        scene_dims = scene_maxes - scene_mins + 1

        # Create height map of the scene
        height_map = np.zeros((scene_dims[0], scene_dims[1])) + scene_mins[2]
        def calculate_height(coord):
            height_map[coord[0], coord[1]] = max(coord[2], height_map[coord[0], coord[1]])
        mapped_coords = coords - [scene_mins[0], scene_mins[1], 0]
        [calculate_height(coord) for coord in mapped_coords]

        # Apply a max smoothing to the height map to fill holes
        filled_height_map = ndimage.maximum_filter(height_map, size=5)  # magic number with 2cm vx size = 10cm real

        # Add sample
        for sample in samples:
            sample_cat = self.id2cat_name[sample]
            cat_path = instance_folder + sample_cat
            file = cat_path + '/' + random.choice(os.listdir(cat_path))
            inst_coords, inst_feats, inst_labels, instance_ids, _ = self.load_ply_w_path(file, scene_name)

            # Add augmentation with attributes
            if self.config.instance_augmentation is not None:
                inst_labels = np.hstack((inst_labels[:, None], np.zeros_like(inst_labels)[:, None]))  # have it with all zeros, we will augment in latent space after encoding
            if self.config.instance_augmentation == 'raw':
                inst_coords, inst_feats, inst_labels = self.augment_instances(inst_coords, inst_feats, inst_labels, instance_ids) # do the augmentation here

            # Voxelize instance too
            inst_coords, inst_feats, inst_labels, inst_vox_transform = self.voxelizer.voxelize(
                inst_coords, inst_feats, inst_labels)

            # Get instance dimensions
            sample_maxes = np.amax(inst_coords, axis=0)
            sample_mins = np.amin(inst_coords, axis=0)
            sample_dims = sample_maxes - sample_mins + 1

            # Start finding a suitable location
            centroid = np.zeros(3, dtype=int)
            iter_num = 0
            while iter_num < self.config.max_instance_placing_iterations:

                # Add random BB
                random_x = random.randint(scene_mins[0], scene_maxes[0])
                random_y = random.randint(scene_mins[1], scene_maxes[1])
                height = float(filled_height_map[random_x - scene_mins[0], random_y - scene_mins[1]]) + 0  # Add some margin (+20cm)
                centroid = np.array([random_x, random_y, int(height + sample_dims[2] / 2.)])

                random_bb = np.array([centroid - (sample_dims / 2.0), centroid + (sample_dims / 2.0)])

                # Check intersection
                is_intersects = False
                for bb_dict in scene_bbs['instances']:
                    # Load and ransform BB
                    bb = np.copy(bb_dict['bb'])
                    homo_bb = np.hstack((bb, np.ones((bb.shape[0], 1), dtype=coords.dtype)))
                    bb = homo_bb @ voxel_scale.T[:, :3]

                    if box_intersect(bb, random_bb):
                        is_intersects = True
                        break

                if not is_intersects:
                    break

                iter_num += 1

            # Push point cloud to BB location
            inst_coords = inst_coords - np.mean(inst_coords, axis=0).astype(int) + centroid

            # Append new inputs
            coords = np.concatenate((coords, inst_coords))
            feats = np.concatenate((feats, inst_feats))
            labels = np.concatenate((labels, inst_labels))

        # Finally, apply rotation augmentation
        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ trans_rot.T[:, :3])

        # Quantize again to remove touching points
        _, unique_map = ME.utils.sparse_quantize(coords_aug, return_index=True, ignore_label=self.config.ignore_label)
        coords_aug, feats, labels = coords_aug[unique_map], feats[unique_map], labels[unique_map]

        return coords_aug, feats, labels

    def augment_instances(self, coords, feats, labels, instance_ids=None):

        # To use precomputed inds we can only update the final scene cloud in the ends
        augmented_coords = []
        augmented_feats = []
        augmented_labels = []
        inds_to_remove = []

        scene_scale = np.array([(coords[:, 0].max() - coords[:, 0].min()),
                               (coords[:, 1].max() - coords[:, 1].min()),
                               (coords[:, 2].max() - coords[:, 2].min())])

        scene_cats = np.unique(labels)
        mapped_cats = np.vectorize(lambda x: self.label_map[x])(scene_cats)
        tail_cats = self.frequency_organized_cats[:, 2].cpu().numpy()
        common_cats = self.frequency_organized_cats[:, 1].cpu().numpy()
        pc_indexes = np.arange(coords.shape[0])

        for label_id in mapped_cats:
            # Only do for valid tail categories
            if label_id != self.ignore_mask and tail_cats[label_id]:

                # point inds for certain category
                cat_inds = labels[:, 0] == self.inverse_label_map[label_id]

                # Do this for scene instance augmentation
                if instance_ids is not None:
                    scene_instances = np.unique(instance_ids[cat_inds])
                    for inst in scene_instances:
                        p_inds = cat_inds * (instance_ids == inst)
                        inst_coords, inst_feats, inst_labels = coords[p_inds], feats[p_inds], labels[p_inds]

                        # Shift color with probability
                        if random.random() < self.aug_color_prob:
                            inst_coords, inst_feats, inst_labels = self.instance_augmentation_transform.shift_color(inst_coords, inst_feats, inst_labels)

                        # Shift scale with probability
                        elif random.random() < self.aug_scale_prob:
                            inst_coords, inst_feats, inst_labels = self.instance_augmentation_transform.shift_scale(inst_coords, inst_feats, inst_labels, scene_scale)

                        augmented_coords += [inst_coords]
                        augmented_feats += [inst_feats]
                        augmented_labels += [inst_labels]
                        inds_to_remove += [pc_indexes[p_inds][:, None]]

                else:  # sampled tail inst
                    inst_coords, inst_feats, inst_labels = coords, feats, labels

                    # Shift color with probability
                    if random.random() < self.aug_color_prob:
                        inst_coords, inst_feats, inst_labels = self.instance_augmentation_transform.shift_color(inst_coords, inst_feats, inst_labels)

                    # Shift scale with probability
                    elif random.random() < self.aug_scale_prob:
                        inst_coords, inst_feats, inst_labels = self.instance_augmentation_transform.shift_scale(inst_coords, inst_feats, inst_labels, scene_scale)

                    augmented_coords += [inst_coords]
                    augmented_feats += [inst_feats]
                    augmented_labels += [inst_labels]
                    inds_to_remove += [pc_indexes[:, None]]

        # Augmentation happened, update the clouds
        if len(augmented_coords) > 0:
            augmented_coords = np.vstack(augmented_coords)
            augmented_feats = np.vstack(augmented_feats)
            augmented_labels = np.vstack(augmented_labels)
            inds_to_remove = np.vstack(inds_to_remove).flatten()

            coords = np.delete(coords, inds_to_remove, axis=0)
            feats = np.delete(feats, inds_to_remove, axis=0)
            labels = np.delete(labels, inds_to_remove, axis=0)

            coords = np.vstack((coords, augmented_coords))
            feats = np.vstack((feats, augmented_feats))
            labels = np.vstack((labels, augmented_labels))

        return coords, feats, labels

    def __getitem__(self, index):

        coords, feats, labels, instance_ids, scene_name = self.load_ply(index)
        scene_name = scene_name.split('/')[-1].split('.')[0]

        # Downsample the pointcloud with finer voxel size before transformation for memory and speed
        if self.PREVOXELIZATION_VOXEL_SIZE is not None:
            _, inds = ME.utils.sparse_quantize(coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
            coords = coords[inds]
            feats = feats[inds]
            labels = labels[inds]

        # Prevoxel transformations
        if self.prevoxel_transform is not None:
            coords, feats, labels = self.prevoxel_transform(coords, feats, labels)

        # Instance based augmentation
        if self.config.instance_augmentation is not None and self.augment_data:
            labels = np.hstack((labels[:, None], np.zeros_like(labels)[:, None]))
            if self.config.instance_augmentation == 'raw':
                coords, feats, labels = self.augment_instances(coords, feats, labels, instance_ids)

        # Add balanced instances
        if self.config.sample_tail_instances and self.augment_data:
            # Don't augment yet, but apply when everything is given
            coords, feats, labels, transformations = self.voxelizer.voxelize(coords, feats, labels, augment=False)
            coords, feats, labels = self.add_instances_to_cloud(coords, feats, labels, scene_name, transformations)
        else:
            # Voxelize as usual
            coords, feats, labels, transformations = self.voxelizer.voxelize(coords, feats, labels)

        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords, feats, labels = self.input_transform(coords, feats, labels)
        if self.target_transform is not None:
            coords, feats, labels = self.target_transform(coords, feats, labels)
        if self.IGNORE_LABELS is not None:
            mapper = lambda x: self.label_map[x]
            if labels.ndim == 1:
                labels = np.vectorize(mapper)(labels)
            else:
                labels[:, 0] = np.vectorize(mapper)(labels[:, 0])

        # Use coordinate features if config is set
        if self.AUGMENT_COORDS_TO_FEATS:
            coords, feats, labels = self._augment_coords_to_feats(coords, feats, labels)

        return_args = [coords, feats, labels, scene_name]

        if self.return_transformation:
            return_args.append(transformations[1].astype(np.float32))

        return tuple(return_args)

    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def get_classids(self):
        return self.VALID_CLASS_IDS

    def get_classnames(self):
        return self.CLASS_LABELS

    def _augment_locfeat(self, pointcloud):
        # Assuming that pointcloud is xyzrgb(...), append location feat.
        pointcloud = np.hstack(
            (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
             pointcloud[:, 6:]))
        return pointcloud

    def test_pointcloud(self, pred_dir, num_labels):

        print('Running full pointcloud evaluation.')
        eval_path = os.path.join(pred_dir, 'fulleval')
        os.makedirs(eval_path, exist_ok=True)
        # Join room by their area and room id.
        # Test independently for each room.
        sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
        hist = np.zeros((num_labels, num_labels))
        for i, data_path in enumerate(self.data_paths):
            room_id = self.get_output_id(i)
            pred = np.load(glob.glob(pred_dir + '/*pred*%04d.npy' % i)[0])

            # Scale with voxel size
            pred[:, :3] *= self.voxelizer.voxel_size

            # pred = np.load(os.path.join(pred_dir, 'pred_%04d_%02d.npy' % (i, 0)))

            # save voxelized pointcloud predictions
            save_point_cloud(
                np.hstack((pred[:, :3], np.array([self.SCANNET_COLOR_MAP[i] for i in pred[:, -1]]))),
                f'{eval_path}/{room_id}_voxel.ply',
                verbose=False)

            fullply_f = self.data_root / data_path
            query_pointcloud = read_plyfile(fullply_f)
            query_xyz = query_pointcloud[:, :3]
            query_label = query_pointcloud[:, -1]
            # Run test for each room.
            pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
            _, result = pred_tree.query(query_xyz)
            ptc_pred = pred[result, 3].astype(int)
            # Save prediciton in txt format for submission.
            np.savetxt(f'{eval_path}/{room_id}.txt', ptc_pred, fmt='%i')
            # Save prediciton in colored pointcloud for visualization.
            save_point_cloud(
                np.hstack((query_xyz, np.array([self.SCANNET_COLOR_MAP[i] for i in ptc_pred]))),
                f'{eval_path}/{room_id}.ply',
                verbose=False)
            # Evaluate IoU.
            if self.IGNORE_LABELS is not None:
                ptc_pred = np.array([self.label_map[x] for x in ptc_pred], dtype=np.int)
                query_label = np.array([self.label_map[x] for x in query_label], dtype=np.int)
            hist += fast_hist(ptc_pred, query_label, num_labels)
        ious = per_class_iu(hist) * 100
        print('mIoU: ' + str(np.nanmean(ious)) + '\n'
                                                 'Class names: ' + ', '.join(self.CLASS_LABELS) + '\n'
                                                                                                  'IoU: ' + ', '.join(
            np.round(ious, 2).astype(str)))

class ScannetVoxelization2cmDataset(ScannetVoxelizationDataset):
    VOXEL_SIZE = 0.02

class Scannet200VoxelizationDataset(ScannetVoxelizationDataset):
    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_200
    CLASS_LABELS = CLASS_LABELS_200
    VALID_CLASS_IDS = VALID_CLASS_IDS_200

    NUM_LABELS = max(SCANNET_COLOR_MAP_LONG.keys()) + 1
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))


class Scannet200Voxelization2cmDataset(Scannet200VoxelizationDataset):
    VOXEL_SIZE = 0.02


