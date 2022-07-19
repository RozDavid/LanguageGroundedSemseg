import collections

import numpy as np
import MinkowskiEngine as ME
from scipy.linalg import expm, norm
from scipy.spatial import KDTree

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

    def __init__(self,
                 voxel_size=1,
                 clip_bound=None,
                 use_augmentation=False,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):
        """
        Args:
          voxel_size: side length of a voxel
          clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
          scale_augmentation_bound: None or (0.9, 1.1)
          rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
            Use random order of x, y, z to prevent bias.
          translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
          ignore_label: label assigned for ignore (not a training label).
        """
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    def get_transformation_matrix(self):
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1 / self.voxel_size
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None, trans_aug_ratio=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        lim = self.clip_bound

        if isinstance(self.clip_bound, (int, float)):
            if bound_size.max() < self.clip_bound:
                return None
            else:
                clip_inds = ((coords[:, 0] >= (-lim + center[0])) & \
                             (coords[:, 0] < (lim + center[0])) & \
                             (coords[:, 1] >= (-lim + center[1])) & \
                             (coords[:, 1] < (lim + center[1])) & \
                             (coords[:, 2] >= (-lim + center[2])) & \
                             (coords[:, 2] < (lim + center[2])))
                return clip_inds

        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) & \
                     (coords[:, 0] < (lim[0][1] + center[0])) & \
                     (coords[:, 1] >= (lim[1][0] + center[1])) & \
                     (coords[:, 1] < (lim[1][1] + center[1])) & \
                     (coords[:, 2] >= (lim[2][0] + center[2])) & \
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None, augment=True, num_pairs=1, dropout_ratio=0.3, dropout_patch_point_num=30):
        # Check input shape for feats and points
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        assert num_pairs == 1 or num_pairs == 2

        # Clip to smaller size if requested
        if self.clip_bound is not None:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            clip_inds = self.clip(coords, center, trans_aug_ratio)
            if clip_inds is not None:
                coords, feats = coords[clip_inds], feats[clip_inds]
                if labels is not None:
                    labels = labels[clip_inds]

        unique_maps = []
        augmented_coords = []
        transforms = []
        for i in range(num_pairs):

            # Get rotation and scale
            M_v, M_r = self.get_transformation_matrix()
            # Apply transformations
            rigid_transformation = M_v
            if augment and self.use_augmentation:
                rigid_transformation = M_r @ rigid_transformation

            homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
            coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

            # key = self.hash(coords_aug)  # floor happens by astype(np.uint64)
            _, unique_map = ME.utils.sparse_quantize(coords_aug, return_index=True, ignore_label=self.ignore_label)

            augmented_coords += [coords_aug]
            unique_maps += [unique_map]
            transforms += [(M_v, M_r)]

        if num_pairs == 1:  # Simply return if simple input is required
            return coords_aug[unique_map], feats[unique_map], labels[unique_map], (M_v, M_r)

        else:  # Magic indexing for finding correspondences
            u_map0 = unique_maps[0]
            u_map1 = unique_maps[1]
            aug_cord0 = augmented_coords[0]
            aug_cord1 = augmented_coords[1]

            # Create boolean umap as well for better intersection calculation
            b_umap0 = np.zeros(coords.shape[0]).astype(bool)
            b_umap0[u_map0] = True
            b_umap1 = np.zeros(coords.shape[0]).astype(bool)
            b_umap1[u_map1] = True

            # Iterate over unique cats and find closest pairs
            corrs0 = np.zeros(b_umap0.sum())
            corrs1 = np.zeros(b_umap1.sum())
            indexer0 = np.arange(b_umap0.sum())
            indexer1 = np.arange(b_umap1.sum())

            utargets = np.unique(labels)
            for target in utargets:
                # Find indices as an intersection with voxelized index map and the target cats
                target_inds0 = indexer0[labels[b_umap0] == target]
                target_inds1 = indexer1[labels[b_umap1] == target]

                # Build tree for targets only
                tree0 = KDTree(coords[target_inds0])
                tree1 = KDTree(coords[target_inds1])

                # Query the other trees for closes correspondence (in most cases this will be the same point itself)
                _, target_corrs0 = tree1.query(coords[target_inds0], k=1, workers=4)
                _, target_corrs1 = tree0.query(coords[target_inds1], k=1, workers=4)

                # Update the global correspondence map
                corrs0[target_inds0] = target_inds1[target_corrs0]
                corrs1[target_inds1] = target_inds0[target_corrs1]


            # Apply unique indexing
            aug_cord0, feats0, labels0, transforms0 = aug_cord0[u_map0], feats[u_map0], labels[u_map0], transforms[0]
            aug_cord1, feats1, labels1, transforms1 = aug_cord1[u_map1], feats[u_map1], labels[u_map1], transforms[1]

            # Dropout patches with given percentage of both augmented clouds
            if dropout_ratio > 0:
                # create trees for nn search
                tree0 = KDTree(aug_cord0)
                tree1 = KDTree(aug_cord1)

                indexer0 = np.arange(aug_cord0.shape[0])
                indexer1 = np.arange(aug_cord1.shape[0])

                # calculate number of seeds for patch_centroids
                seed_num0 = round(aug_cord0.shape[0] * dropout_ratio / dropout_patch_point_num)
                seed_num1 = round(aug_cord1.shape[0] * dropout_ratio / dropout_patch_point_num)
                cloud0_patch_seeds = np.random.choice(indexer0, size=seed_num0, replace=False)
                cloud1_patch_seeds = np.random.choice(indexer1, size=seed_num1, replace=False)

                # Select the nearest points around seeds
                _, drop_inds0 = tree0.query(aug_cord0[cloud0_patch_seeds], k=dropout_patch_point_num, workers=4)
                _, drop_inds1 = tree1.query(aug_cord1[cloud1_patch_seeds], k=dropout_patch_point_num, workers=4)

                # Remove duplicated points
                drop_inds0 = np.unique(drop_inds0.flatten())
                drop_inds1 = np.unique(drop_inds1.flatten())

                # delete these inds from the batch
                drop_mask0 = np.ones(aug_cord0.shape[0]).astype(bool)
                drop_mask0[drop_inds0] = False
                drop_mask1 = np.ones(aug_cord1.shape[0]).astype(bool)
                drop_mask1[drop_inds1] = False

                aug_cord0, feats0, labels0, corrs0 = aug_cord0[drop_mask0], feats0[drop_mask0], labels0[drop_mask0], corrs0[drop_mask0].astype(int)
                aug_cord1, feats1, labels1, corrs1 = aug_cord1[drop_mask1], feats1[drop_mask1], labels1[drop_mask1], corrs1[drop_mask1].astype(int)

                shift_by_deleted_points_in_0 = np.zeros(drop_mask0.shape[0])
                shift_by_deleted_points_in_0[0] = (1 - drop_mask0[0])
                for i in range(1, len(shift_by_deleted_points_in_0)):
                    shift_by_deleted_points_in_0[i] = shift_by_deleted_points_in_0[i-1] + (1 - drop_mask0[i])

                shift_by_deleted_points_in_1 = np.zeros(drop_mask1.shape[0])
                shift_by_deleted_points_in_1[0] = (1 - drop_mask1[0])
                for i in range(1, len(shift_by_deleted_points_in_1)):
                    shift_by_deleted_points_in_1[i] = shift_by_deleted_points_in_1[i - 1] + (1 - drop_mask1[i])

                # map the values
                corrs0 = np.vectorize(lambda corr0: corr0 - shift_by_deleted_points_in_1[corr0])(corrs0)
                corrs1 = np.vectorize(lambda corr1: corr1 - shift_by_deleted_points_in_0[corr1])(corrs1)

            return (aug_cord0, feats0, labels0, transforms0, corrs0), \
                   (aug_cord1, feats1, labels1, transforms1, corrs1)


    def voxelize_temporal(self,
                          coords_t,
                          feats_t,
                          labels_t,
                          centers=None,
                          return_transformation=False):
        # Legacy code, remove
        if centers is None:
            centers = [
                          None,
                      ] * len(coords_t)
        coords_tc, feats_tc, labels_tc, transformation_tc = [], [], [], []

        # ######################### Data Augmentation #############################
        # Get rotation and scale
        M_v, M_r = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_v
        if self.use_augmentation:
            rigid_transformation = M_r @ rigid_transformation
        # ######################### Voxelization #############################
        # Voxelize coords
        for coords, feats, labels, center in zip(coords_t, feats_t, labels_t, centers):

            ###################################
            # Clip the data if bound exists
            if self.clip_bound is not None:
                trans_aug_ratio = np.zeros(3)
                if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                    for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                        trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

                clip_inds = self.clip(coords, center, trans_aug_ratio)
                if clip_inds is not None:
                    coords, feats = coords[clip_inds], feats[clip_inds]
                    if labels is not None:
                        labels = labels[clip_inds]
            ###################################

            homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
            coords_aug = np.floor(homo_coords @ rigid_transformation.T)[:, :3]

            coords_aug, feats, labels = ME.utils.sparse_quantize(
                coords_aug, feats, labels=labels, ignore_label=self.ignore_label)

            coords_tc.append(coords_aug)
            feats_tc.append(feats)
            labels_tc.append(labels)
            transformation_tc.append(rigid_transformation.flatten())

        return_args = [coords_tc, feats_tc, labels_tc]
        if return_transformation:
            return_args.append(transformation_tc)

        return tuple(return_args)


def test():
    N = 16575
    coords = np.random.rand(N, 3) * 10
    feats = np.random.rand(N, 4)
    labels = np.floor(np.random.rand(N) * 3)
    coords[:3] = 0
    labels[:3] = 2
    voxelizer = Voxelizer()
    print(voxelizer.voxelize(coords, feats, labels))


if __name__ == '__main__':
    test()
