import datetime
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import hyperparams
from layers import InputLayer, ConvLayerWithReLU, PoolingLayer, FullyConnectedLayerWithReLU, \
    FullyConnectedLayerWithSoftmax, NetworkBuilder, BuildingBlock

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

y = (
    NetworkBuilder()
    .add_layer(InputLayer("input", x_image))
    .add_layer(BuildingBlock(
        "block1", 1, 32, [ConvLayerWithReLU("conv1", 1, 32, 5, 1), PoolingLayer("pool1", 32, tf.nn.max_pool, 2, 2)]
    ))
    .add_layer(ConvLayerWithReLU("conv2", 32, 64, 5, 1))
    .add_layer(PoolingLayer("pool2", 64, tf.nn.max_pool, 2, 2))
    .add_layer(FullyConnectedLayerWithReLU("fc1", 7 * 7 * 64, 1024))
    .add_layer(FullyConnectedLayerWithSoftmax("softmax", 1024, 10))
).build()

with tf.name_scope('train'):
    cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('summaries'):
    train_error = tf.scalar_summary("training-error", 1 - accuracy)
    merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('summaries/mnist_conv_arch', sess.graph_def)

sess.run(tf.initialize_all_variables())
for i in xrange(hyperparams.TRAINING_STEPS):
    batch = mnist.train.next_batch(50)
    feed = {x: batch[0], y_: batch[1]}
    sess.run(train_step, feed_dict=feed)
    result = sess.run(merged, feed_dict=feed)
    writer.add_summary(result, i)


print 'test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels
}, session=sess)
