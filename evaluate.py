# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_eval.py
import math
import os
import time
from datetime import datetime as dt

import numpy
import tensorflow as tf
from tensorflow.python.platform import gfile

from config import FLAGS
from util import extract_global_step


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

        last = None
        while True:
            last = _eval_once(last, saver, read_checkpoint_path, summary_writer, top_k_op, summary_op, test_err)
            if FLAGS.run_once or last == FLAGS.training_steps:
                break
            time.sleep(FLAGS.eval_interval_secs)


def _eval_once(last, saver, read_checkpoint_path, summary_writer, top_k_op, summary_op, test_err):
    with tf.Session() as sess:
        # restore training progress
        global_step, ckpt = _has_new_checkpoint(read_checkpoint_path, last)
        if ckpt:
            saver.restore(sess, os.path.join(read_checkpoint_path, ckpt))
            print '%s - Found new checkpoint file from step %i.' % (dt.now(), global_step)
        else:
            print '%s - No new checkpoint file found.' % dt.now()
            return global_step

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print '%s - Started computing test error for step %i.' % (dt.now(), global_step)
        try:
            num_iter = int(math.ceil(FLAGS.max_num_examples) / FLAGS.batch_size)
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(top_k_op)
                true_count += numpy.sum(predictions)
                step += 1

            accuracy = true_count / total_sample_count
            test_error = 1 - accuracy
            print '%s - step %i: test error = %.2f%%' % (dt.now(), global_step, test_error * 100)

            summary = sess.run(summary_op, feed_dict={test_err: test_error})
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        # return the current global step to track the progress
        return global_step


# since tf.train.get_checkpoint_state always returns the latest checkpoint file, but we only want to run the script
# when a new checkpoint is available
def _has_new_checkpoint(path, last):
    new_files = {}
    for f in gfile.ListDirectory(path):
        if not gfile.IsDirectory(f):
            try:
                global_step = extract_global_step(f)
            except Exception:  # pylint: disable=broad-except
                continue
            if global_step and (not last or global_step > last):
                new_files[global_step] = f
    if new_files == {}:
        return last, None
    min_global_step = min(new_files)
    return min_global_step, new_files[min_global_step]


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
