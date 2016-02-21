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
            if x.get_shape()[1] != f.get_shape()[1]:
                x = self._adjust_dimensions(x, x.get_shape(), f.get_shape(), self._name)
            b = bias_variable([self._out_channels], name=self._name + '_bias', initial=0.0)
            return tf.nn.relu(f + x + b, name=self._name + 'ResidualReLU')


def _identity_mapping(x, x_shape, f_shape, name):
    # spatial resolution reduction using a simulated 1x1 max-pooling with stride 2
    x = tf.nn.max_pool(_mask_input(x), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, f_shape[3].value - x_shape[3].value]], name=name + '_identityMap')


def _projection_mapping(x, x_shape, f_shape, name):
    # TODO this is ugly. Replace with 1x1 convolution as soon as it's supported.
    # convolution-like feature extraction using 1x1 max-pooling with stride 2
    extracted = tf.nn.max_pool(_mask_input(x), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    w = unoptimized_weight_variable([x_shape[3].value, f_shape[3].value], name=name + '_residualWeights')

    # simulate 1x1 convolution using batch matrix multiplication
    extracted_shape = extracted.get_shape().as_list()
    # stack all the extracted patches of the image
    patch_stack = tf.reshape(extracted, [extracted_shape[0], -1, extracted_shape[3]])
    # make W the same batch size as the input by simply replicating it
    w_expanded = tf.tile(tf.expand_dims(w, 0), [extracted_shape[0], 1, 1])
    # final matrix multiplication - should apply the operation specified in the documentation of conv2d:
    # "3. For each patch, right-multiplies the filter matrix and the image patch vector."
    projection = tf.batch_matmul(patch_stack, w_expanded)
    # make the result a 2D image batch again
    return tf.reshape(projection, [extracted_shape[0], extracted_shape[1], extracted_shape[2], -1])


def _mask_input(x):
    x_shape = x.get_shape().as_list()
    mask = [[row % 2 == 0 and column % 2 == 0 for column in xrange(x_shape[2])] for row in xrange(x_shape[1])]
    mask = tf.cast(tf.constant(mask, dtype=tf.bool), tf.float32)

    mask = tf.expand_dims(tf.expand_dims(mask, 0), 3)
    mask = tf.tile(mask, [x_shape[0], 1, 1, x_shape[3]])

    return tf.mul(x, mask)
