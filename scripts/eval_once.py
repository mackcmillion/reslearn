import math
import numpy

import tensorflow as tf
from datetime import datetime as dt

from config import FLAGS
from datasets.cifar10 import Cifar10
from models.resnet6nplus2 import CIFAR10ResNet20, CIFAR10ResNet32, CIFAR10ResNet44, CIFAR10ResNet56

tf.app.flags.DEFINE_string('checkpoint', '/home/max/Studium/Kurse/BA2/results/checkpoints/resnet_56_optB_2016-04-24_15-18-01/cifar10-resnet-56.ckpt-64000',
                           """The checkpoint to compute test error from.""")


def evaluate(dataset, model):

    dataset.pre_graph()

    with tf.Graph().as_default():
        # input and evaluation procedure
        images, true_labels = dataset.evaluation_inputs()
        predictions = model.inference(images, dataset.num_classes, False)
        top_k_op = _in_top_k(predictions, true_labels)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            _eval_once(sess, coord, saver, tf.app.flags.FLAGS.checkpoint, top_k_op, dataset)

            coord.request_stop()
            coord.join(threads)


def _eval_once(sess, coord, saver, read_checkpoint_path, top_k_op, dataset):
    saver.restore(sess, read_checkpoint_path)
    print '%s - Restored model from checkpoint %s.' % (dt.now(), read_checkpoint_path)

    try:
        num_iter = int(math.ceil((1.0 * 10000) / FLAGS.batch_size))
        true_count = 0
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run(top_k_op)
            true_count += numpy.sum(predictions)
            step += 1
            print '%s - %d of %d images: test error = %.4f' % \
                  (dt.now(), step * FLAGS.batch_size, 10000, 1 - ((true_count * 1.0) / (step * FLAGS.batch_size)))

        accuracy = (true_count * 1.0) / total_sample_count
        test_error = 1 - accuracy
        print '\n%s - final test error over %d images: test error = %.4f%%' % \
              (dt.now(), total_sample_count, test_error * 100)

    except Exception as e:  # pylint: disable=broad-except
        print e
        coord.request_stop(e)


def _in_top_k(predictions, true_labels):
    # softmax is not necessary here
    # predictions = tf.nn.softmax(predictions, name='eval_softmax')
    return tf.nn.in_top_k(predictions, true_labels, FLAGS.top_k)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate(Cifar10(), CIFAR10ResNet56())


if __name__ == '__main__':
    tf.app.run()
