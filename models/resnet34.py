import tensorflow as tf

from architecture.building_blocks import add_n_conv3x3_blocks
from architecture.layers import NetworkBuilder, ConvLayer, PoolingLayer, InputLayer, FullyConnectedLayer
from model import ResidualModel


class ResNet34(ResidualModel):

    def __init__(self):
        super(ResNet34, self).__init__('resnet-34', ['imagenet', 'yelp'])

    def inference(self, x, num_classes):
        builder = NetworkBuilder()

        (builder
         .add_layer(InputLayer('input', x, 3))
         .add_layer(ConvLayer('conv1', 3, 64, filter_size=7, stride=2))
         .add_layer(PoolingLayer('max_pool', 64, tf.nn.max_pool, filter_size=3, stride=2))
         )

        add_n_conv3x3_blocks(builder, 3, 64, 64, self._adjust_dimensions, 'conv2')
        add_n_conv3x3_blocks(builder, 4, 64, 128, self._adjust_dimensions, 'conv3')
        add_n_conv3x3_blocks(builder, 6, 128, 256, self._adjust_dimensions, 'conv4')
        add_n_conv3x3_blocks(builder, 3, 256, 512, self._adjust_dimensions, 'conv5')

        (builder
         .add_layer(PoolingLayer('avg_pool', 512, tf.nn.avg_pool, filter_size=3, stride=1))
         # this last layer has no softmax since training and evaluation handle softmax internally
         .add_layer(FullyConnectedLayer('fc', 7 * 7 * 512, num_classes))
         )

        return builder.build()
