# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_eval.py
import time

import math
import tensorflow as tf
from datetime import datetime as dt

from config import FLAGS


def evaluate(dataset, model, summary_path, read_checkpoint_path):
    with tf.Graph().as_default():
        # input and evaluation procedure
        images, true_labels = dataset.evaluation_inputs()
        predictions = model.inference(images, dataset.num_classes)
        top_k_op = _in_top_k(predictions, true_labels)

        saver = tf.train.Saver(tf.trainable_variables())

        test_err = tf.placeholder(tf.float32, shape=[], name='test_err')
        # FIXME moving average not working right now since we are using different sessions
        # test_err_avg_op = _add_validation_error_summary(val_err)
        _add_test_error_summary(test_err)

        # with tf.control_dependencies([val_err_avg_op]):
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(summary_path, tf.get_default_graph().as_graph_def())

        while True:
            finished = _eval_once(saver, read_checkpoint_path, summary_writer,
                                                   top_k_op, summary_op, test_err)
            if FLAGS.run_once or finished:
                break
            time.sleep(FLAGS.eval_interval_secs)


def _eval_once(saver, read_checkpoint_path, summary_writer, top_k_op, summary_op, test_err):
    with tf.Session() as sess:
        # restore training progress
        ckpt = tf.train.get_checkpoint_state(read_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print '%s - Found new checkpoint file from step %i.' % (dt.now(), global_step)
        else:
            print '%s - No new checkpoint file found.' % dt.now()
            return False

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            num_iter = int(math.ceil(FLAGS.max_num_examples) / FLAGS.batch_size)
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(top_k_op)
                true_count += sess.run(tf.reduce_sum(tf.cast(predictions, tf.int32)))
                step += 1

            accuracy = true_count / total_sample_count
            test_error = 1 - accuracy
            print '%s - test error = %.3f' % (dt.now(), test_error)

            summary = sess.run(summary_op, feed_dict={test_err: test_error})
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        # the last checkpoint is always the one with the total number of training epochs as step
        return global_step == FLAGS.training_epochs


def _in_top_k(predictions, true_labels):
    predictions = tf.nn.softmax(predictions, name='eval_softmax')
    return tf.nn.in_top_k(predictions, true_labels, FLAGS.top_k)


def _top_k_10crop(predictions, true_labels):
    predictions = tf.nn.softmax(predictions, name='eval_softmax')
    pred_mean = tf.reduce_mean(predictions, reduction_indices=0)
    pred_mean = tf.expand_dims(pred_mean, 0)
    true_labels = tf.expand_dims(true_labels, 0)
    return tf.nn.in_top_k(pred_mean, true_labels, FLAGS.top_k)


def _add_test_error_summary(test_err):
    # test_err_avg = tf.train.ExponentialMovingAverage(0.9, name='test_err_avg')
    # test_err_avg_op = test_err_avg.apply([test_err])

    tf.scalar_summary('test_error_raw', test_err)
    # tf.scalar_summary('test_error_averaged', test_err_avg.average(test_err))

    # return test_err_avg_op
