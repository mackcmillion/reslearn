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
    w = weight_variable(shape=[ksize, ksize, x.get_shape()[3].value, out_channels],
                        name=name + '_weights',
                        stddev=stddev_init,
                        wd=FLAGS.weight_decay)

    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME', name=name)

    if relu:
        x = batch_normalize(x, out_channels, phase_train, name)
        # b = bias_variable([out_channels],
        #                   name=name + '_bias',
        #                   initial=0.0)
        x = tf.nn.relu(x
                       # + b
                       , name=name + '_ReLU')

    return x


def pooling_layer(x, pooling_func, ksize, stride, name):
    return pooling_func(
            x,
            ksize=[1, ksize, ksize, 1],
            strides=[1, stride, stride, 1],
            padding='SAME',
            name=name
    )


def fc_layer(x, out_channels, activation_fn, name):
    n_hat = x.get_shape()[1].value
    stddev_init = math.sqrt(2.0 / n_hat)
    w = weight_variable([x.get_shape()[1].value, out_channels],
                        name=name + '_weights',
                        stddev=stddev_init,
                        wd=FLAGS.weight_decay)
    b = bias_variable([out_channels],
                      name=name + '_bias',
                      initial=0.0)

    x = tf.matmul(x, w, name=name) + b

    if activation_fn:
        x = activation_fn(x, name=name + '_activation')

    return x


# batch normalization according to
# S. Ioffe - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
# def batch_normalize(x, out_channels, phase_train, name):
#     mean, variance = tf.nn.moments(x, [0, 1, 2])
#     beta = tf.Variable(tf.zeros([out_channels]), name=name + '_beta', trainable=True)
#     gamma = tf.Variable(tf.truncated_normal([out_channels]), name=name + '_gamma', trainable=True)
#     return tf.nn.batch_norm_with_global_normalization(x, mean, variance, beta, gamma, 0.001,
#                                                       scale_after_normalization=True, name=name + '_batchNorm')
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


def weight_variable(shape, name, stddev, wd):
    initial = tf.random_normal(shape, stddev=stddev)
    var = tf.Variable(initial_value=initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name + '_weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(shape, name, initial=0.0):
    initial = tf.constant(initial, shape=shape)
    return tf.Variable(initial, name=name)
