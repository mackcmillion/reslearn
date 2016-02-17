import tensorflow as tf

from layers import Layer, NetworkBuilder
from util import bias_variable, unoptimized_weight_variable


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
    def __init__(self, name, in_channels, out_channels, layers, adjust_dimensions='IDENTITY'):
        super(ResidualBuildingBlock, self).__init__(name, in_channels, out_channels, layers)
        assert adjust_dimensions == 'IDENTITY' or adjust_dimensions == 'PROJECTION'
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
    # FIXME this is actually not correct... don't use 2x2 pooling
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, f_shape[3].value - x_shape[3].value]], name=name + '_identityMap')


def _projection_mapping(x, x_shape, f_shape, name):
    # FIXME using 1x1 convolution with stride 2 makes TensorFlow throw exception
    w = unoptimized_weight_variable([2, 2, x_shape[3].value, f_shape[3].value], name=name + '_residualWeights')
    stride = int(x_shape[2].value / f_shape[2].value)
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME', name=name + '_residualProjection')
    # print x_shape, f_shape
    # x_shape = x_shape.as_list()
    # f_shape = f_shape.as_list()
    # assert 2 * f_shape[1] == x_shape[1]
    # assert 2 * f_shape[2] == x_shape[2]
    # assert 2 * x_shape[3] == f_shape[3]
    #
    # # simulate 1x1 convolution with stride 2
    # x = tf.reshape(x, [-1, x_shape[1]])
    # x = tf.pack(tf.unpack(x)[0::2])
    # x = tf.reshape(x, [-1])
    # x = tf.pack(tf.unpack(x)[0::2])
    # x = tf.reshape(x, [-1, f_shape[1], f_shape[2], x_shape[3]])
    #
    # print x_shape, f_shape
    #
    # w = unoptimized_weight_variable([x_shape[3], f_shape[3]], name=name + '_residualWeights')
    # res = tf.matmul(x, w)
    # print x_shape, f_shape, res.get_shape()
    # return res
