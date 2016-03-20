import tensorflow as tf

from architecture.layers import conv_layer, bias_variable, batch_normalize


def residual_building_block(x, to_wrap, adjust_dimensions, name):

    if x.get_shape() != to_wrap.get_shape():
        if adjust_dimensions == 'IDENTITY':
            x = _identity_mapping(x, x.get_shape(), to_wrap.get_shape(), name)
        elif adjust_dimensions == 'PROJECTION':
            x = _projection_mapping(x, x.get_shape(), to_wrap.get_shape(), name)
        else:
            raise ValueError('Unknown adjust dimensions strategy.')

    # this is the residual addition
    x += to_wrap

    # TODO optionally drop ReLU here
    return tf.nn.relu(x, name=name + '_ResidualReLU')


def _identity_mapping(x, x_shape, f_shape, name):
    # spatial resolution reduction using a simulated 1x1 max-pooling with stride 2
    x = tf.nn.max_pool(_mask_input(x), [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, f_shape[3].value - x_shape[3].value]], name=name + '_identityMap')


def _projection_mapping(x, x_shape, f_shape, name):
    # TODO this is ugly. Replace with 1x1 convolution with stride 2 as soon as it's supported.
    # extracted = tf.nn.max_pool(_mask_input(x), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    # w = weight_variable([1, 1, x_shape[3].value, f_shape[3].value], name=name + '_residualWeights',
    #                     # FIXME n_hat may be wrong
    #                     n_hat=x_shape[0].value * x_shape[1].value * x_shape[2].value,
    #                     wd=FLAGS.weight_decay)
    # return tf.nn.conv2d(extracted, w, [1, 1, 1, 1], padding='VALID')
    return x


def _mask_input(x):
    x_shape = x.get_shape().as_list()
    mask = [[row % 2 == 0 and column % 2 == 0 for column in xrange(x_shape[2])] for row in xrange(x_shape[1])]
    mask = tf.cast(tf.constant(mask, dtype=tf.bool), tf.float32)

    mask = tf.expand_dims(tf.expand_dims(mask, 0), 3)
    mask = tf.tile(mask, [x_shape[0], 1, 1, x_shape[3]])

    return tf.mul(x, mask)


def conv3x3_block(x, in_channels, out_channels, adjust_dimensions, namespace, phase_train):
    assert x.get_shape()[3].value == in_channels
    if in_channels != out_channels:
        assert out_channels == 2 * in_channels
        stride = 2
    else:
        stride = 1

    with tf.name_scope(namespace):
        f = conv_layer(x, out_channels, ksize=3, relu=True, stride=stride, phase_train=phase_train,
                       name=namespace + '_1')
        f = conv_layer(f, out_channels, ksize=3, relu=False, stride=1, phase_train=phase_train, name=namespace + '_2')

        y = residual_building_block(x, to_wrap=f, adjust_dimensions=adjust_dimensions, name=namespace)
    return y


def add_n_conv3x3_blocks(x, n, in_channels, out_channels, adjust_dimensions, namespace, phase_train):
    assert n > 0
    # add first 3x3 layer that maybe performs downsampling
    x = conv3x3_block(x, in_channels, out_channels, adjust_dimensions, namespace + '_1', phase_train)
    # add the rest n-1 layers that keep dimensions
    for i in xrange(1, n):
        x = conv3x3_block(x, out_channels, out_channels, adjust_dimensions, namespace + ('_%i' % (i + 1)), phase_train)
    return x
