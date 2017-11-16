import functools

import tensorflow as tf

from dpipe.config import register
from .base import ModelCore
from .layers import spatial_batch_norm, prelu, nearest_neighbour

activation = functools.partial(prelu, feature_dims=[1])


def cb(t, n_chans, kernel_size, training, name):
    with tf.variable_scope(name):
        t = tf.layers.conv3d(t, n_chans, kernel_size, use_bias=False, data_format='channels_first')
        t = spatial_batch_norm(t, training=training, data_format='channels_first')
        return t


def cba(t, n_chans, kernel_size, training, name):
    with tf.variable_scope(name):
        return activation(cb(t, n_chans, kernel_size, training, 'cb'))


def build_path(t, blocks, kernel_size, training, name):
    with tf.variable_scope(name):
        for i, n_chans in enumerate(blocks):
            t = cba(t, n_chans, kernel_size, training, f'cba_{i}_a')
            t = cba(t, n_chans, kernel_size, training, f'cba_{i}_b')
        return t


downsampling_ops = {
    'average': lambda x: tf.layers.average_pooling3d(x, 3, 3, 'same', 'channels_first'),
    'sampling': lambda x: x[:, :, 1:-1:3, 1:-1:3, 1:-1:3]
}


def build_model(t_det_in, t_con_in, kernel_size, n_classes, training, name, *,
                path_blocks=(30, 40, 40, 50), n_chans_com=150, dropout):
    with tf.variable_scope(name):
        with tf.variable_scope("to_restore"):
            t_det = build_path(t_det_in, path_blocks, kernel_size, training, 'detailed')
            t_con = build_path(t_con_in, path_blocks, kernel_size, training, 'context')
            t_con = nearest_neighbour(t_con, 3, data_format='channels_first', name='upsample')

            t_com = tf.concat([t_con, t_det], axis=1)
            t_com = cba(dropout(t_com), n_chans_com, 1, training, name='comm_1')
            t_com = cba(dropout(t_com), n_chans_com, 1, training, name='comm_2')
        return cb(t_com, n_classes, 1, training, 'C')


@register('deepmedic_els')
class DeepMedicEls(ModelCore):
    def __init__(self, *, n_chans_in, n_chans_out,  downsampling_type='sampling', with_dropout=True):
        super().__init__(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        self.kernel_size = 3
        self.with_dropout = with_dropout
        self.downsampling_op = downsampling_ops[downsampling_type]

    def build(self, training_ph):
        def dropout(x):
            if self.with_dropout:
                return tf.layers.dropout(x, training=training_ph, noise_shape=(tf.shape(x)[0], tf.shape(x)[1], 1, 1, 1))
            else:
                return x

        input_shape = (None, self.n_chans_in, None, None, None)
        x_det_ph = tf.placeholder(tf.float32, input_shape, name='x_det')
        x_con_ph = tf.placeholder(tf.float32, input_shape, name='x_con')

        x_con = self.downsampling_op(x_con_ph)

        logits = build_model(x_det_ph, x_con, self.kernel_size, self.n_chans_out, training_ph, name='deep_medic',
                             dropout=dropout)

        return [x_det_ph, x_con_ph], logits
