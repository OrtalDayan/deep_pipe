import numpy as np

from dpipe.medim.divide import compute_n_parts_per_axis
from dpipe.medim.shape_utils import compute_shape_from_spatial
from dpipe.medim.patch import pad
from .patch_3d import Patch3DPredictor, spatial_dims


def pad_spatial_size(x, spatial_size: np.array, spatial_dims):
    padding = np.zeros((len(x.shape), 2), dtype=int)
    padding[spatial_dims, 1] = spatial_size - np.array(x.shape)[list(spatial_dims)]
    return pad(x, padding, np.min(x, axis=spatial_dims, keepdims=True))


def slice_spatial_size(x, spatial_size, spatial_dims):
    slices = np.array([slice(None)] * len(x.shape))
    slices[list(spatial_dims)] = list(map(slice, [0] * len(spatial_size), spatial_size))
    return x[tuple(slices)]


def find_fixed_spatial_size(spatial_size, spatial_patch_size):
    return compute_n_parts_per_axis(spatial_size, spatial_patch_size) * spatial_patch_size


class Patch3DFixedPredictor(Patch3DPredictor):
    def divide_x(self, x):
        spatial_size = np.array(x.shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_patch_size)
        x_padded = pad_spatial_size(x, fixed_spatial_size, spatial_dims)
        return super().divide_x(x_padded)

    def divide_y(self, y):
        spatial_size = np.array(y.shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_patch_size)
        y_padded = pad_spatial_size(y, fixed_spatial_size, spatial_dims)
        return super().divide_y(y_padded)

    def combine_y(self, y_parts, x_shape):
        spatial_size = np.array(x_shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_patch_size)
        y_pred = super().combine_y(y_parts, compute_shape_from_spatial(x_shape, fixed_spatial_size, spatial_dims))
        y_pred = slice_spatial_size(y_pred, spatial_size, spatial_dims)
        return y_pred
