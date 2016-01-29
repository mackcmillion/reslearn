from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from layers import InputLayer, ConvLayerWithReLU, PoolingLayer, FullyConnectedLayerWithReLU, \
    FullyConnectedLayerWithSoftmax

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])

input_layer = InputLayer("input", x_image)
conv1 = ConvLayerWithReLU("conv1", 1, 32, input_layer, 5, 1)
pool1 = PoolingLayer("pool1", 32, conv1, tf.nn.max_pool, 2, 2)
conv2 = ConvLayerWithReLU("conv2", 32, 64, pool1, 5, 1)
pool2 = PoolingLayer("pool2", 64, conv2, tf.nn.max_pool, 2, 2)
fc1 = FullyConnectedLayerWithReLU("fc1", 7 * 7 * 64, 1024, pool2)
output_layer = FullyConnectedLayerWithSoftmax("softmax", 1024, 10, fc1)

y = output_layer.eval()

cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in xrange(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]
        }, session=sess)
        print 'step %d, training accuracy %g' % (i, train_accuracy)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

print 'test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels
}, session=sess)
