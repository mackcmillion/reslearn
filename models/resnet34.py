import tensorflow as tf

from architecture.building_blocks import add_n_conv3x3_blocks
from architecture.layers import NetworkBuilder, ConvLayer, PoolingLayer, InputLayer, FullyConnectedLayer
from model import ResidualModel


class ResNet34(ResidualModel):

    def __init__(self):
        super(ResNet34, self).__init__('resnet-34', ['imagenet'])

    def inference(self, x, num_classes, phase_train):
        builder = NetworkBuilder()

        (builder
         .add_layer(InputLayer('input', x, 3, phase_train))
         .add_layer(ConvLayer('conv1', 3, 64, filter_size=7, stride=2, phase_train=phase_train))
         .add_layer(PoolingLayer('max_pool', 64, tf.nn.max_pool, filter_size=3, stride=2, phase_train=phase_train))
         )

        add_n_conv3x3_blocks(builder, 3, 64, 64, self._adjust_dimensions, 'conv2', phase_train)
        add_n_conv3x3_blocks(builder, 4, 64, 128, self._adjust_dimensions, 'conv3', phase_train)
        add_n_conv3x3_blocks(builder, 6, 128, 256, self._adjust_dimensions, 'conv4', phase_train)
        add_n_conv3x3_blocks(builder, 3, 256, 512, self._adjust_dimensions, 'conv5', phase_train)

        (builder
         .add_layer(PoolingLayer('avg_pool', 512, tf.nn.avg_pool, filter_size=3, stride=1, phase_train=phase_train))
         # this last layer has no softmax since training and evaluation handle softmax internally
         .add_layer(FullyConnectedLayer('fc', 7 * 7 * 512, num_classes, phase_train))
         )

        return builder.build()
