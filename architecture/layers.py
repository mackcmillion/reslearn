import math

import tensorflow as tf

from config import FLAGS


# optimized weight variable initialization according to
# K. He - Delving Deep into Rectifiers: Surpassing Human Performance in ImageNet Classification
# where n_hat = k**2 * d
# with k the kernel size (k x k) and d the number of output channels
def conv_layer(x, out_channels, ksize, relu, stride, phase_train, name):
    n_hat = ksize * ksize * out_channels
    stddev_init = math.sqrt(2.0 / (1.0 * n_hat))
    w = weight_variable(shape=[ksize, ksize, x.get_shape()[3].value, out_channels],
                        name=name + '_weights',
                        stddev=stddev_init,
                        wd=FLAGS.weight_decay)

    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME', name=name)
    x = batch_normalize(x, out_channels, phase_train, name)

    if relu:
        x = tf.nn.relu(x, name=name + '_ReLU')

    return x


def pooling_layer(x, pooling_func, ksize, stride, name):
    return pooling_func(
            x,
            ksize=[1, ksize, ksize, 1],
            strides=[1, stride, stride, 1],
            padding='VALID',
            name=name
    )


def fc_layer(x, out_channels, activation_fn, name):
    stddev_init = 1.0 / math.sqrt(1.0 * out_channels)
    w = weight_variable([x.get_shape()[1].value, out_channels],
                        name=name + '_weights',
                        stddev=stddev_init, uniform=True,
                        wd=FLAGS.weight_decay)
    b = bias_variable([out_channels],
                      name=name + '_bias',
                      initial=0.0,
                      wd=FLAGS.weight_decay)

    x = tf.matmul(x, w, name=name) + b

    if activation_fn:
        x = activation_fn(x, name=name + '_activation')

    return x


# batch normalization according to
# S. Ioffe - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
def batch_normalize(x, out_channels, phase_train, name):
    mean, variance = tf.nn.moments(x, [0, 1, 2])
    beta = tf.Variable(tf.constant(0.0, shape=[out_channels]), name=name + '_beta',
                       trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[out_channels]), name=name + '_gamma',
                        trainable=True)
    return tf.nn.batch_norm_with_global_normalization(x, mean, variance, beta, gamma, 0.001,
                                                      scale_after_normalization=True, name=name + '_batchNorm')


def weight_variable(shape, name, stddev, wd, uniform=False):
    if uniform:
        initial = tf.random_uniform(shape, -stddev, stddev)
    else:
        initial = tf.random_normal(shape, stddev=stddev)
    var = tf.Variable(initial_value=initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name + '_weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(shape, name, initial, wd):
    initial = tf.constant(initial, shape=shape)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name + '_bias_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable_random_init(shape, name, stddev, uniform=False):
    if uniform:
        initial = tf.random_uniform(shape, -stddev, stddev)
    else:
        initial = tf.random_normal(shape, stddev=stddev)
    var = tf.Variable(initial_value=initial, name=name)
    return var
