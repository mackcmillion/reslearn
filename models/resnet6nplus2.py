import tensorflow as tf
from abc import ABCMeta

from architecture.building_blocks import add_n_conv3x3_blocks
from architecture.layers import NetworkBuilder, InputLayer, ConvLayer, PoolingLayer, FullyConnectedLayer
from models.model import ResidualModel


class ResNet6nplus2(ResidualModel):
    __metaclass__ = ABCMeta

    def __init__(self, name, n):
        super(ResNet6nplus2, self).__init__(name, ['cifar10'])
        self._n = n
        # in the paper, they only use identity mapping when testing on CIFAR-10
        self._adjust_dimensions = 'IDENTITY'

    def inference(self, x, num_classes, phase_train):
        builder = NetworkBuilder()

        (builder
         .add_layer(InputLayer('input', x, 3, phase_train))
         .add_layer(ConvLayer('conv1', 3, 16, filter_size=3, stride=1, phase_train=phase_train))
         )

        add_n_conv3x3_blocks(builder, 2 * self._n, 16, 16, self._adjust_dimensions, 'conv2', phase_train)
        add_n_conv3x3_blocks(builder, 2 * self._n, 16, 32, self._adjust_dimensions, 'conv3', phase_train)
        add_n_conv3x3_blocks(builder, 2 * self._n, 32, 64, self._adjust_dimensions, 'conv4', phase_train)

        (builder
         .add_layer(PoolingLayer('avg_pool', 64, tf.nn.avg_pool, filter_size=3, stride=1, phase_train=phase_train))
         # this last layer has no softmax since training and evaluation handle softmax internally
         .add_layer(FullyConnectedLayer('fc', 8 * 8 * 64, num_classes, phase_train))
         )

        return builder.build()


class CIFAR10ResNet20(ResNet6nplus2):
    def __init__(self):
        super(CIFAR10ResNet20, self).__init__('cifar10-resnet-20', 3)


class CIFAR10ResNet32(ResNet6nplus2):
    def __init__(self):
        super(CIFAR10ResNet32, self).__init__('cifar10-resnet-32', 5)


class CIFAR10ResNet44(ResNet6nplus2):
    def __init__(self):
        super(CIFAR10ResNet44, self).__init__('cifar10-resnet-44', 7)


class CIFAR10ResNet56(ResNet6nplus2):
    def __init__(self):
        super(CIFAR10ResNet56, self).__init__('cifar10-resnet-56', 9)


class CIFAR10ResNet110(ResNet6nplus2):
    def __init__(self):
        super(CIFAR10ResNet110, self).__init__('cifar10-resnet-110', 18)


class CIFAR10ResNet1202(ResNet6nplus2):
    def __init__(self):
        super(CIFAR10ResNet1202, self).__init__('cifar10-resnet-1202', 200)
