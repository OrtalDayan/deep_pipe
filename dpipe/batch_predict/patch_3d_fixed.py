import numpy as np

from dpipe.config import register
from dpipe.medim.divide import compute_n_parts_per_axis
from dpipe.medim.shape_utils import compute_shape_from_spatial
from dpipe.medim.utils import pad
from dpipe.medim.features import get_coordinate_features
from .patch_3d import Patch3DPredictor, spatial_dims
from .utils import validate_object, predict_object


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


class Patch3DVolume(Patch3DFixedPredictor):
    def divide_y(self, y):
        return [x.sum().astype('float32') for x in super().divide_y(y)]

    def combine_y(self, y_parts, x_shape):
        return sum(y_parts)


class Patch3DFixedQuantilesPredictor(Patch3DFixedPredictor):
    def __init__(self, x_patch_sizes: list, y_patch_size: list, n_quantiles: int, padding_mode: str):
        super().__init__(x_patch_sizes=x_patch_sizes, y_patch_size=y_patch_size,
                         padding_mode=padding_mode)
        self.n_quantiles = n_quantiles

    def divide_x(self, x):
        xs_parts = super().divide_x(x)
        quantiles = np.percentile(x, np.linspace(0, 100, self.n_quantiles))
        xs_parts.append(len(xs_parts[0]) * [quantiles])
        return xs_parts


class Patch3DFixedQuantilesPredictor(Patch3DFixedQuantilesPredictor):
    def divide_x(self, x):
        *xs_parts, quantiles = super().divide_x(x)
        shape = np.array(x.shape)[list(spatial_dims)]
        coordinates = [get_coordinate_features(shape, shape // 2, self.y_patch_size)] * len(xs_parts[0])
        return (*xs_parts, coordinates, quantiles)


class Patch3DFixedDropoutPredictor(Patch3DFixedPredictor):
    def __init__(self, x_patch_sizes: list, y_patch_size: list, padding_mode: str = 'min', n: int = 10):
        super().__init__(x_patch_sizes=x_patch_sizes, y_patch_size=y_patch_size,
                         padding_mode=padding_mode)
        self.n = n

    def validate(self, x, y, *, validate_fn):
        xs_parts = self.divide_x(x)
        y_parts_true = self.divide_y(y)

        assert len(xs_parts[0]) == len(y_parts_true)

        temp_y_pred = []
        temp_loss = []
        for i in range(self.n):
            y_preds, loss = validate_object(zip(*xs_parts, y_parts_true), validate_fn=validate_fn)
            temp_y_pred.append(y_preds)
            temp_loss.append(loss)

        y_preds = np.array(temp_y_pred).mean(axis=0)
        # mean of losses is higer than loss of mean for logloss
        loss = np.array(temp_loss).mean(axis=0)

        return self.combine_y(y_preds, x.shape), loss

    def predict(self, x, *, predict_fn):
        xs_parts = self.divide_x(x)
        y_preds = np.array([predict_object(zip(*xs_parts), predict_fn=predict_fn) for i in range(self.n)]).mean(axis=0)

        return self.combine_y(y_preds, x.shape)
