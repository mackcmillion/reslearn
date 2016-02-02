import tensorflow as tf

from layers import Layer, NetworkBuilder
from util import bias_variable, unoptimized_weight_variable, batch_normalize


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
    def __init__(self, name, in_channels, out_channels, layers, adjust_dimensions='PROJECTION'):
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
            if x.get_shape() != f.get_shape():
                x = self._adjust_dimensions(x, x.get_shape(), f.get_shape(), self._name)
            b = bias_variable([self._out_channels], name=self._name + '_bias', initial=0.0)
            # TODO not sure if to apply batch normalization to the residual or only to f
            residual = f + x
            return tf.nn.relu(
                    batch_normalize(residual, self._out_channels, self._name) + b,
                    name=self._name + 'ResidualReLU')


def _identity_mapping(x, x_shape, f_shape, name):
    return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, f_shape[3].value - x_shape[3].value]], name=name + '_identityMap')


def _projection_mapping(x, x_shape, f_shape, name):
    # FIXME using 1x1 convolution with stride 2 makes TensorFlow throw exception
    w = unoptimized_weight_variable([2, 2, x_shape[3].value, f_shape[3].value], name=name + '_residualWeights')
    stride = int(f_shape[3].value / x_shape[3].value)
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME', name=name + '_residualProjection')
