import tensorflow as tf
from abc import ABCMeta

from architecture.building_blocks import add_n_conv3x3_blocks
from architecture.layers import conv_layer, pooling_layer, fc_layer
from config import FLAGS
from models.model import ResidualModel


class ResNet6nplus2(ResidualModel):
    __metaclass__ = ABCMeta

    def __init__(self, name, n):
        super(ResNet6nplus2, self).__init__(name, ['cifar10'])
        self._n = n
        # however, in the paper they only use identity mapping when testing on CIFAR-10
        assert FLAGS.adjust_dimensions_strategy in ['A', 'B']
        if FLAGS.adjust_dimensions_strategy == 'A':
            self._adjust_dimensions = 'IDENTITY'
        else:
            self._adjust_dimensions = 'PROJECTION'

    def inference(self, x, num_classes, phase_train):
        x = conv_layer(x, 16, ksize=3, relu=True, stride=1, name='conv1', phase_train=phase_train)
        x = add_n_conv3x3_blocks(x, 2 * self._n, 16, 16, self._adjust_dimensions, 'conv2', phase_train)
        x = add_n_conv3x3_blocks(x, 2 * self._n, 16, 32, self._adjust_dimensions, 'conv3', phase_train)
        x = add_n_conv3x3_blocks(x, 2 * self._n, 32, 64, self._adjust_dimensions, 'conv4', phase_train)
        x = pooling_layer(x, tf.nn.avg_pool, ksize=8, stride=1, name='avg_pool')
        x = fc_layer(tf.squeeze(x), num_classes, activation_fn=None, name='fc')

        return x


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
