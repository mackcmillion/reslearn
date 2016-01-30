import tensorflow as tf
from abc import ABCMeta, abstractmethod


class NetworkBuilder(object):

    def __init__(self, layer_before=None):
        self._current_layer = layer_before

    def add_layer(self, layer):
        layer.layer_before = self._current_layer
        self._current_layer = layer
        return self

    def build(self):
        return self._current_layer.eval()


class Layer(object):

    __metaclass__ = ABCMeta

    _layer_before = None

    def __init__(self, name, in_channels, out_channels):
        self._name = name
        self._in_channels = in_channels
        self._out_channels = out_channels

    @abstractmethod
    def eval(self):
        pass

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def layer_before(self):
        return self._layer_before

    @layer_before.setter
    def layer_before(self, layer):
        self._layer_before = layer


class BuildingBlock(Layer):

    def __init__(self, name, in_channels, out_channels, layers):
        super(BuildingBlock, self).__init__(name, in_channels, out_channels)
        self._layers = layers

    def eval(self):
        with tf.name_scope(self._name):
            builder = NetworkBuilder(self._layer_before)
            for layer in self._layers:
                builder.add_layer(layer)
            return builder.build()


class InputLayer(Layer):

    def __init__(self, name, x):
        super(InputLayer, self).__init__(name, None, 1)
        self._x = x

    def eval(self):
        return self._x


class ConvLayer(Layer):

    def __init__(self, name, in_channels, out_channels, filter_size, stride):
        super(ConvLayer, self).__init__(name, in_channels, out_channels)
        self._filter_size = filter_size
        self._stride = stride

    def eval(self):
        x = self.layer_before.eval()
        w = _weight_variable(
                [self._filter_size, self._filter_size, self._in_channels, self._out_channels],
                name=self._name + '_weights'
        )
        return tf.nn.conv2d(x, w, strides=[1, self._stride, self._stride, 1], padding='SAME', name=self._name)


class ConvLayerWithReLU(ConvLayer):

    def eval(self):
        b = _bias_variable([self._out_channels], name=self._name + 'ReLU_bias')
        return tf.nn.relu(super(ConvLayerWithReLU, self).eval() + b, name=self._name + 'ReLU')


class PoolingLayer(Layer):

    def __init__(self, name, channels, pooling_func, filter_size, stride):
        super(PoolingLayer, self).__init__(name, channels, channels)
        self._pooling_func = pooling_func
        self._filter_size = filter_size
        self._stride = stride

    def eval(self):
        x = self.layer_before.eval()
        return self._pooling_func(
                x,
                ksize=[1, self._filter_size, self._filter_size, 1],
                strides=[1, self._stride, self._stride, 1],
                padding='SAME',
                name=self._name
        )


class FullyConnectedLayer(Layer):

    def __init__(self, name, in_channels, out_channels):
        super(FullyConnectedLayer, self).__init__(name, in_channels, out_channels)

    def eval(self):
        x = self.layer_before.eval()
        if self._in_channels != self.layer_before.out_channels:
            x = tf.reshape(x, [-1, self._in_channels])
        w = _weight_variable([self._in_channels, self.out_channels], name=self._name + '_weights')
        return tf.matmul(x, w, name=self._name)


class FullyConnectedLayerWithReLU(FullyConnectedLayer):

    def eval(self):
        b = _bias_variable([self._out_channels], name=self._name + 'ReLU_bias')
        return tf.nn.relu(super(FullyConnectedLayerWithReLU, self).eval() + b, name=self._name + 'ReLU')


class FullyConnectedLayerWithSoftmax(FullyConnectedLayer):

    def eval(self):
        b = _bias_variable([self._out_channels], name=self._name + 'Softmax_bias')
        return tf.nn.softmax(super(FullyConnectedLayerWithSoftmax, self).eval() + b, name=self._name + 'Softmax')


def _weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def _bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
