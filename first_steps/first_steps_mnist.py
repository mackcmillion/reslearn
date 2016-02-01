from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_error = tf.scalar_summary("training-error", accuracy)
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('summaries', sess.graph_def)

sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    feed = {x: batch[0], y_: batch[1]}
    sess.run(train_step, feed_dict=feed)
    if i % 10 == 0:
        result = sess.run(merged, feed_dict=feed)
        writer.add_summary(result, i)

print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}, session=sess)
