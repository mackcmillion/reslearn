import tensorflow as tf
from architecture.building_blocks import ResidualBuildingBlock

from architecture.layers import NetworkBuilder, ConvLayer, PoolingLayer, ConvLayerWithReLU, \
    InputLayer, FullyConnectedLayer


def resnet_34(x):

    builder = NetworkBuilder()

    (builder
     .add_layer(InputLayer('input', x, 3))
     .add_layer(ConvLayer('conv1', 3, 64, filter_size=7, stride=2))
     .add_layer(PoolingLayer('max_pool', 64, tf.nn.max_pool, filter_size=3, stride=2))
     )

    for i in xrange(3):
        namespace = 'conv2_' + str(i + 1)
        builder.add_layer(ResidualBuildingBlock(
                namespace, 64, 64,
                [
                    ConvLayerWithReLU(namespace + '_1', 64, 64, filter_size=3, stride=1),
                    ConvLayer(namespace + '_2', 64, 64, filter_size=3, stride=1)
                ]
        ))

    for i in xrange(4):
        namespace = 'conv3_' + str(i + 1)
        builder.add_layer(ResidualBuildingBlock(
                namespace, 64 if i == 0 else 128, 128,
                [
                    ConvLayerWithReLU(namespace + '_1', 64 if i == 0 else 128, 128, filter_size=3,
                                      stride=2 if i == 0 else 1),
                    ConvLayer(namespace + '_2', 128, 128, filter_size=3, stride=1)
                ]
        ))

    for i in xrange(6):
        namespace = 'conv4_' + str(i + 1)
        builder.add_layer(ResidualBuildingBlock(
                namespace, 128 if i == 0 else 256, 256,
                [
                    ConvLayerWithReLU(namespace + '_1', 128 if i == 0 else 256, 256, filter_size=3,
                                      stride=2 if i == 0 else 1),
                    ConvLayer(namespace + '_2', 256, 256, filter_size=3, stride=1)
                ]
        ))

    for i in xrange(3):
        namespace = 'conv5_' + str(i + 1)
        builder.add_layer(ResidualBuildingBlock(
                namespace, 256 if i == 0 else 512, 512,
                [
                    ConvLayerWithReLU(namespace + '_1', 256 if i == 0 else 512, 512, filter_size=3,
                                      stride=2 if i == 0 else 1),
                    ConvLayer(namespace + '_2', 512, 512, filter_size=3, stride=1)
                ]
        ))

    (builder
     .add_layer(PoolingLayer('avg_pool', 512, tf.nn.avg_pool, filter_size=3, stride=1))
     # this last layer has no softmax since training and evaluation handle softmax internally
     .add_layer(FullyConnectedLayer('fc', 7 * 7 * 512, 1000))
     )

    return builder.build()
