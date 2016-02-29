import tensorflow as tf

from architecture.layers import ConvLayerWithReLU, ConvLayer
from config import FLAGS
from layers import Layer, NetworkBuilder
from util import bias_variable, unoptimized_weight_variable, weight_variable


class BuildingBlock(Layer):
    def __init__(self, name, in_channels, out_channels, layers):
        super(BuildingBlock, self).__init__(name, in_channels, out_channels)
        self._layers = layers

    def _eval(self):
        with tf.name_scope(self._name):
            return self._eval_aux()

    def _eval_aux(self):
        builder = NetworkBuilder(self._layer_before)
        for layer in self._layers:
            builder.add_layer(layer)
        return builder.build()


class ResidualBuildingBlock(BuildingBlock):
    def __init__(self, name, in_channels, out_channels, layers, adjust_dimensions):
        super(ResidualBuildingBlock, self).__init__(name, in_channels, out_channels, layers)
        assert adjust_dimensions == 'IDENTITY' or adjust_dimensions == 'PROJECTION', \
            'Unknown adjusting strategy %s' % adjust_dimensions
        if adjust_dimensions == 'IDENTITY':
            self._adjust_dimensions = _identity_mapping
        else:
            self._adjust_dimensions = _projection_mapping

    def _eval(self):
        x = self._layer_before.eval()
        with tf.name_scope(self._name):
            f = self._eval_aux()
            if x.get_shape()[1] != f.get_shape()[1]:
                x = self._adjust_dimensions(x, x.get_shape(), f.get_shape(), self._name)
            b = bias_variable([self._out_channels], name=self._name + '_bias', initial=0.0)
            return tf.nn.relu(f + x + b, name=self._name + 'ResidualReLU')


def _identity_mapping(x, x_shape, f_shape, name):
    # spatial resolution reduction using a simulated 1x1 max-pooling with stride 2
    x = tf.nn.max_pool(_mask_input(x), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, f_shape[3].value - x_shape[3].value]], name=name + '_identityMap')


def _projection_mapping(x, x_shape, f_shape, name):
    # TODO this is ugly. Replace with 1x1 convolution with stride 2 as soon as it's supported.
    extracted = tf.nn.max_pool(_mask_input(x), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    w = weight_variable([1, 1, x_shape[3].value, f_shape[3].value], name=name + '_residualWeights',
                        n_hat=x_shape[0].value * x_shape[1].value * x_shape[2].value,
                        wd=FLAGS.weight_decay)
    return tf.nn.conv2d(extracted, w, [1, 1, 1, 1], padding='SAME')


def _mask_input(x):
    x_shape = x.get_shape().as_list()
    mask = [[row % 2 == 0 and column % 2 == 0 for column in xrange(x_shape[2])] for row in xrange(x_shape[1])]
    mask = tf.cast(tf.constant(mask, dtype=tf.bool), tf.float32)

    mask = tf.expand_dims(tf.expand_dims(mask, 0), 3)
    mask = tf.tile(mask, [x_shape[0], 1, 1, x_shape[3]])

    return tf.mul(x, mask)


def conv3x3_block(in_channels, out_channels, adjust_dimensions, namespace):
    if in_channels != out_channels:
        assert out_channels == 2 * in_channels
        stride = 2
    else:
        stride = 1
    return ResidualBuildingBlock(
            namespace, in_channels, out_channels,
            [
                ConvLayerWithReLU(namespace + '_1', in_channels, out_channels, filter_size=3,
                                  stride=stride),
                ConvLayer(namespace + '_2', out_channels, out_channels, filter_size=3, stride=1)
            ],
            adjust_dimensions=adjust_dimensions
    )


def add_n_conv3x3_blocks(builder, n, in_channels, out_channels, adjust_dimensions, namespace):
    assert n > 0
    # add first 3x3 layer that maybe performs downsampling
    builder.add_layer(conv3x3_block(in_channels, out_channels, adjust_dimensions, namespace + '_1'))
    # add the rest n-1 layers that keep dimensions
    for i in xrange(1, n):
        builder.add_layer(conv3x3_block(out_channels, out_channels, adjust_dimensions, namespace + ('_%i' % (i + 1))))
