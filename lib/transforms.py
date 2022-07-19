import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import matplotlib
import torch
import open3d as o3d

import MinkowskiEngine as ME


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class ChromaticTranslation(object):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels, corrs=None):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)

        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


class ChromaticAutoContrast(object):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels, corrs=None):
        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)
            assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

            scale = 255 / (hi - lo)

            contrast_feats = (feats[:, :3] - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats

        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


class ChromaticJitter(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, labels, corrs=None):
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


class ChromaticScale(object):

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def __call__(self, coords, feats, labels, corrs=None):

        feats[:, :3] = feats[:, :3] * self.scale_factor

        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


class HueSaturationTranslation(object):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels, corrs=None):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


##############################
# Coordinate transformations
##############################
class RandomDropout(object):

    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, coords, feats, labels, corrs=None):
        if random.random() < self.dropout_ratio:
            N = len(coords)
            inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)

            coords = coords[inds]
            feats = feats[inds]
            labels = labels[inds]
            if corrs is not None:
                corrs = corrs[inds]

        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis, is_temporal):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels, corrs=None):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]

        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


class ElasticDistortion:

    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords += interp(coords) * magnitude
        return coords, feats, labels

    def __call__(self, coords, feats, labels, corrs=None):
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity,
                                                                    magnitude)

        if corrs is None:
            return coords, feats, labels
        else:
            return coords, feats, labels, corrs


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


##############################
# Coordinate transformations
##############################
class InstanceAugmentation(object):

    def __init__(self, config):

        self.config = config

        self.rgb_to_hsv = HueSaturationTranslation.rgb_to_hsv
        self.hsv_to_rgb = HueSaturationTranslation.hsv_to_rgb

        # Color parameters
        self.color_shifts = ['Red', 'Green', 'Blue', 'Yellow', 'Dark', 'Bright']
        self.red_hue = 0. / 360.0
        self.yellow_hue = 60. / 360.0
        self.green_hue = 120. / 360.0
        self.blue_hue = 240. / 360.0
        self.white_scale = 2.0
        self.black_scale = 1. / self.white_scale

        # Scale parameters
        self.size_shifts = [0.5, 1.5]


    def shift_hue(self, colors, h_out):
        hsv = self.rgb_to_hsv(colors / 255.)
        hsv[..., 0] = h_out
        rgb = self.hsv_to_rgb(hsv) * 255.
        return rgb


    def shift_color(self, coords, feats, labels):
        color_direction = random.sample(self.color_shifts, 1)[0]

        if color_direction == 'Red':
            feats = self.shift_hue(feats, self.red_hue)
            labels[:, 1] = 1
        elif color_direction == 'Green':
            feats = self.shift_hue(feats, self.green_hue)
            labels[:, 1] = 2
        elif color_direction == 'Blue':
            feats = self.shift_hue(feats, self.blue_hue)
            labels[:, 1] = 3
        elif color_direction == 'Yellow':
            feats = self.shift_hue(feats, self.yellow_hue)
            labels[:, 1] = 4
        elif color_direction == 'Dark':
            feats = (feats * self.black_scale).astype(int)
            labels[:, 1] = 5
        elif color_direction == 'Bright':
            diff_to_max = 255 - feats
            feats = (255 - (diff_to_max / self.white_scale)).astype(int)
            labels[:, 1] = 6

        return coords, feats, labels


    def shift_scale(self, coords, feats, labels, scene_scale):

        scale_direction = np.random.uniform(low=0., high=2.)

        # Upsample for scaling up
        if scale_direction > 1.:

            # pick random positive scaling factor
            inst_scale = np.array([(coords[:, 0].max() - coords[:, 0].min()),
                                   (coords[:, 1].max() - coords[:, 1].min()),
                                   (coords[:, 2].max() - coords[:, 2].min())])
            scale_direction = np.random.uniform(low=1.0, high=min(self.size_shifts[1], (scene_scale/inst_scale).min()))

            # scale and center to same centroid in X-Y, push up in Z
            center_x = (coords[:, 0].min() + coords[:, 0].max()) / 2.
            center_y = (coords[:, 1].min() + coords[:, 1].max()) / 2.
            min_z = coords[:, 2].min()

            coords *= scale_direction
            coords += np.array([center_x, center_y, min_z]) * (1 - scale_direction)

            labels = np.ones((coords.shape[0], 2)) * labels[0, 0]
            labels[:, 1] = 7
            return coords, feats, labels

        elif scale_direction <= 1.:

            # pick random negative scaling factor
            scale_direction = np.random.uniform(low=self.size_shifts[0], high=1.0)

            # scale and center to same centroid in X-Y, push up in Z
            center_x = (coords[:, 0].min() + coords[:, 0].max()) / 2.
            center_y = (coords[:, 1].min() + coords[:, 1].max()) / 2.
            min_z = coords[:, 2].min()

            coords *= scale_direction
            coords += np.array([center_x, center_y, min_z]) * (1 - scale_direction)

            labels[:, 1] = 8
            return coords, feats, labels


class cfl_collate_fn_factory:
    """Generates collate function for coords, feats, labels.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels, scene_names = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch, scene_names_batch = [], [], [], []

        batch_id = 0
        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                    f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
                )
                break
            coords_batch.append(torch.from_numpy(coords[batch_id]).int())
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            scene_names_batch.append(scene_names[batch_id])

            batch_id += 1

        # Concatenate all lists
        coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
        return coords_batch, feats_batch.float(), labels_batch, scene_names_batch


class cflt_collate_fn_factory:
    """Generates collate function for coords, feats, labels, point_clouds, transformations.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels, scene_names, *transformations = list(zip(*list_data))
        cfl_collate_fn = cfl_collate_fn_factory(limit_numpoints=self.limit_numpoints)
        coords_batch, feats_batch, labels_batch, scene_names_batch = cfl_collate_fn(list(zip(coords, feats, labels, scene_names)))
        num_truncated_batch = coords_batch[:, -1].max().item() + 1

        batch_id = 0
        transformations_batch = []
        for transformation in transformations:
            if batch_id >= num_truncated_batch:
                break
            transformations_batch.append(torch.from_numpy(transformation[0]).float())
            batch_id += 1

        return coords_batch, feats_batch, labels_batch, scene_names_batch, transformations_batch


class paired_cfl_collate_fn_factory:

    """Generates collate function for coords, feats, labels.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):

        pair0, pair1 = list(zip(*list_data))

        max_batch = 10e5  # haha, if only, btw use it for equal batch size if larger than limit point
        collated_pair = []
        for pair in [pair0, pair1]:
            coords_batch, feats_batch, labels_batch, corrs_batch, scene_names_batch = [], [], [], [], []

            batch_num_points = 0
            for batch_id in range(min(len(pair), max_batch)):

                coords, feats, labels, corrs, scene_names = pair[batch_id]

                num_points = coords.shape[0]
                batch_num_points += num_points
                if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                    num_full_points = sum(len(c) for c in coords)
                    num_full_batch_size = len(coords)
                    logging.warning(
                        f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                        f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
                    )
                    max_batch = batch_id
                    break
                coords_batch.append(torch.from_numpy(coords).int())
                feats_batch.append(torch.from_numpy(feats))
                labels_batch.append(torch.from_numpy(labels).int())
                corrs_batch.append(torch.from_numpy(corrs).int())
                scene_names_batch.append(scene_names)

            # Concatenate all lists
            batched_coords, batched_feats, batched_labels = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
            _, batched_corrs = ME.utils.sparse_collate(coords_batch, corrs_batch)
            collated_pair += [(batched_coords, batched_feats.float(), batched_labels, batched_corrs, scene_names_batch)]

        return tuple(collated_pair)
