import math

import tensorflow as tf

from config import FLAGS


# optimized weight variable initialization according to
# K. He - Delving Deep into Rectifiers: Surpassing Human Performance in ImageNet Classification
# where n_hat = k**2 * d
# with k the image size (k x k) and d the number of channels
def conv_layer(x, out_channels, ksize, relu, stride, phase_train, name):
    # TODO optionally adjust that
    n_hat = ((int(x.get_shape()[1].value / stride) ** 2) * out_channels)
    stddev_init = math.sqrt(2.0 / n_hat)

    def activation_fn(x_):
        afn = batch_normalize(x_, out_channels, phase_train=phase_train, name=name)
        if relu:
            return tf.nn.relu(afn)
        return afn

    return tf.contrib.layers.convolution2d(
            x,
            out_channels,
            kernel_size=(ksize, ksize),
            activation_fn=activation_fn,
            stride=(stride, stride),
            weight_init=tf.random_normal_initializer(mean=0.0, stddev=stddev_init),
            bias_init=tf.constant_initializer(0.0),
            name=name,
            weight_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
    )


def pooling_layer(x, pooling_func, ksize, stride, name):
    return pooling_func(
            x,
            ksize=[1, ksize, ksize, 1],
            strides=[1, stride, stride, 1],
            padding='SAME',
            name=name
    )


def fc_layer(x, out_channels, activation_fn, name):
    return tf.contrib.layers.fully_connected(
            x,
            out_channels,
            activation_fn=activation_fn,
            weight_init=tf.contrib.layers.xavier_initializer(),
            bias_init=tf.constant_initializer(0.0),
            name=name,
            weight_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
    )


# batch normalization according to
# S. Ioffe - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
# code from http://stackoverflow.com/a/34634291/2206976
def batch_normalize(x, out_channels, phase_train, name, scope='bn', affine=True):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[out_channels]), name=name + '_beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_channels]), name=name + '_gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name=name + '_moments')
        ema = tf.train.ExponentialMovingAverage(0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.constant(phase_train, shape=[], dtype=tf.bool),
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, affine)
        return normed
