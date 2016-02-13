from math import sqrt

import tensorflow as tf


def unoptimized_weight_variable(shape, name, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


# optimized weight variable initialization according to
# K. He - Delving Deep into Rectifiers: Surpassing Human Performance in ImageNet Classification
# where n_hat = k**2 * d
# with k the image size (k x k) and d the number of channels
def weight_variable(shape, name, n_hat):
    initial = tf.truncated_normal(shape, stddev=sqrt(n_hat))
    return tf.Variable(initial, name=name)


def bias_variable(shape, name, initial=0.1):
    initial = tf.constant(initial, shape=shape)
    return tf.Variable(initial, name=name)


# batch normalization according to
# S. Ioffe - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
def batch_normalize(x, out_channels, name):
    mean, variance = tf.nn.moments(x, [0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name=name + '_beta')
    gamma = tf.Variable(tf.truncated_normal([out_channels]), name=name + '_gamma')
    return tf.nn.batch_norm_with_global_normalization(x, mean, variance, beta, gamma, 0.001,
                                                      scale_after_normalization=True, name=name + '_batchNorm')


def encode_one_hot(label_batch, num_labels):
    sparse_labels = tf.reshape(label_batch, [-1, 1])
    derived_size = tf.shape(label_batch)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, num_labels])
    return tf.sparse_to_dense(concated, outshape, sparse_values=1.0, default_value=0.0)

