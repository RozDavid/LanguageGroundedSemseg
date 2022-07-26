import numpy as np
import os
from lib.pc_utils import read_plyfile, save_point_cloud

import json
import pandas as pd
from plyfile import PlyData
import open3d as o3d
from lib.constants.dataset_sets import *

def point_indices_from_group(points, seg_indices, group, labels_pd, CLASS_IDs):
    group_segments = np.array(group['segments'])
    label = group['label']

    label_ids = labels_pd[labels_pd['raw_category'] == label]['id']
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

    if not label_id in CLASS_IDs:
        label_id = 0

    # get points, where segindices (points labelled with segment ids) are in the group segment list
    point_IDs = np.where(np.isin(seg_indices, group_segments))

    return points[point_IDs], point_IDs[0], label_id

def save_instance(segment_points, label_id, cat_name, scene_id, base_path, segment_faces=None, limited_annotation_points=None):

    if scene_id in TRAIN_SCENES:
        path = base_path + f'/train/train_instances/{cat_name}'
    elif scene_id in VAL_SCENES:
        path = base_path + f'/train/val_instances/{cat_name}'
    else:
        return

    if not os.path.exists(path):
        os.makedirs(path)

    path, dirs, files = next(os.walk(path))
    file_count = len(files)

    if limited_annotation_points is None:
        labels = np.ones((segment_points.shape[0], 1), dtype=int) * label_id
    else:
        labels = np.zeros((segment_points.shape[0], 1), dtype=int)
        labels[limited_annotation_points] = label_id

    labbelled_instance_points = np.append(segment_points[:, :6], labels, axis=1)

    # Push to origin
    centroid = np.mean(segment_points[:, :3], axis=0)
    labbelled_instance_points[:, :3] -= centroid

    # Save
    out_file = path + f'/{cat_name}_{file_count}.ply'
    save_point_cloud(labbelled_instance_points, out_file, with_label=True, verbose=False, faces=segment_faces)


def load_pcd(path):
    filepath = path
    plydata = PlyData.read(str(filepath))
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array([data['label']], dtype=np.float32).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(feats / 255.)

    return pcd, feats, labels


def box_intersect(box_a, box_b):
    return (box_a[0, 0] <= box_b[1, 0] and box_a[1, 0] >= box_b[0, 0]) and (
            box_a[0, 1] <= box_b[1, 1] and box_a[1, 1] >= box_b[0, 1]) and (
                   box_a[0, 2] <= box_b[1, 2] and box_a[1, 2] >= box_b[0, 2])


def box_contains(parent_box, child_box, inflate_size = 0.):

    parent_min = parent_box[0,:] - inflate_size
    parent_max = parent_box[1,:] + inflate_size

    child_min = child_box[0,:]
    child_max = child_box[1, :]

    return np.all(np.greater(child_min, parent_min)) and np.all(np.less(child_max, parent_max))

def box_contains_percentage_inflate(parent_box, child_box, inflate_size = 0.):

    inflate = np.abs(parent_box[0, :] - parent_box[1, :]) * inflate_size

    parent_min = parent_box[0, :] - inflate
    parent_max = parent_box[1, :] + inflate

    child_min = child_box[0,:]
    child_max = child_box[1, :]

    return np.all(np.greater(child_min, parent_min)) and np.all(np.less(child_max, parent_max))
