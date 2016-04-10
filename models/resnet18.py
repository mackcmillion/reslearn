import tensorflow as tf

from architecture.building_blocks import add_n_conv3x3_blocks
from architecture.layers import conv_layer, pooling_layer, fc_layer
from model import ResidualModel


class ResNet18(ResidualModel):

    def __init__(self):
        super(ResNet18, self).__init__('resnet-18', ['imagenet', 'yelp'])

    def inference(self, x, num_classes, phase_train):
        print x.get_shape()
        x = conv_layer(x, 64, ksize=7, relu=True, stride=2, name='conv1', phase_train=phase_train)
        print x.get_shape()
        x = pooling_layer(x, tf.nn.max_pool, ksize=3, stride=2, name='max_pool', padding='SAME')
        print x.get_shape()

        x = add_n_conv3x3_blocks(x, 2, 64, 64, self._adjust_dimensions, 'conv2', phase_train)
        print x.get_shape()
        x = add_n_conv3x3_blocks(x, 2, 64, 128, self._adjust_dimensions, 'conv3', phase_train)
        print x.get_shape()
        x = add_n_conv3x3_blocks(x, 2, 128, 256, self._adjust_dimensions, 'conv4', phase_train)
        print x.get_shape()
        x = add_n_conv3x3_blocks(x, 2, 256, 512, self._adjust_dimensions, 'conv5', phase_train)
        print x.get_shape()

        x = pooling_layer(x, tf.nn.avg_pool, ksize=7, stride=1, name='avg_pool')
        print x.get_shape()
        x = fc_layer(tf.squeeze(x), num_classes, activation_fn=None, name='fc')
        print x.get_shape()

        return x
