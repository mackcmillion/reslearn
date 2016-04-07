# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_eval.py
import math
import os
import time
from datetime import datetime as dt

import numpy
import tensorflow as tf
from tensorflow.python.platform import gfile

import util
from config import FLAGS
from util import extract_global_step


def evaluate(dataset, model, summary_path, read_checkpoint_path):
    with tf.Graph().as_default():
        # input and evaluation procedure
        images, true_labels = dataset.evaluation_inputs()
        predictions = model.inference_ten_crop(images, dataset.num_classes, 224, False)
        eval_op = dataset.eval_op(predictions, true_labels)
        test_err_op = dataset.test_error

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

        test_err = tf.placeholder(tf.float32, shape=[], name='test_err')
        # FIXME test error averaged starts at 0
        test_err_avg_op = _add_test_error_summary(test_err)

        with tf.control_dependencies([test_err_avg_op]):
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(summary_path, tf.get_default_graph().as_graph_def())

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            last = None
            while True:
                last = _eval_once(sess, coord, last, saver, read_checkpoint_path, summary_writer, eval_op, test_err_op,
                                  summary_op, test_err)
                if FLAGS.run_once or last == FLAGS.training_steps:
                    break
                time.sleep(FLAGS.eval_interval_secs)

            coord.request_stop()
            coord.join(threads)


def _eval_once(sess, coord, last, saver, read_checkpoint_path, summary_writer, eval_op, test_err_op, summary_op,
               test_err):
    # restore training progress
    global_step, ckpt = _has_new_checkpoint(read_checkpoint_path, last)
    if ckpt:
        saver.restore(sess, os.path.join(read_checkpoint_path, ckpt))
        print '%s - Found new checkpoint file from step %i.' % (dt.now(), global_step)
    else:
        print '%s - No new checkpoint file found.' % dt.now()
        return global_step

    print '%s - Started computing test error for step %i.' % (dt.now(), global_step)
    try:
        num_iter = int(math.ceil((1.0 * FLAGS.max_num_examples) / FLAGS.batch_size))
        accumulated = 0.0
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run(eval_op)
            accumulated += numpy.sum(predictions)
            step += 1

        test_error, test_error_name = test_err_op(accumulated, total_sample_count)
        print '%s - step %i: %s = %.2f%%' % (dt.now(), global_step, test_error_name, test_error * 100)

        summary = sess.run(summary_op, feed_dict={test_err: test_error})
        summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
        print e
        coord.request_stop(e)

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
    # softmax is not necessary here
    # predictions = tf.nn.softmax(predictions, name='eval_softmax')
    return tf.nn.in_top_k(predictions, true_labels, FLAGS.top_k)


def _test_error(predictions, true_labels, dataset):
    softmaxed = tf.nn.softmax(predictions)
    correct_prediction = tf.equal(tf.argmax(softmaxed, 1),
                                  tf.argmax(util.encode_one_hot(true_labels, dataset.num_classes), 1))
    return tf.cast(correct_prediction, tf.float32)


def _top_k_10crop(predictions, true_labels):
    predictions = tf.nn.softmax(predictions, name='eval_softmax')
    pred_mean = tf.reduce_mean(predictions, reduction_indices=0)
    pred_mean = tf.expand_dims(pred_mean, 0)
    true_labels = tf.expand_dims(true_labels, 0)
    return tf.nn.in_top_k(pred_mean, true_labels, FLAGS.top_k)


def _add_test_error_summary(test_err):
    test_err_avg = tf.train.ExponentialMovingAverage(0.9, name='test_err_avg')
    test_err_avg_op = test_err_avg.apply([test_err])

    tf.scalar_summary('test_error_raw', test_err)
    tf.scalar_summary('test_error_averaged', test_err_avg.average(test_err))

    return test_err_avg_op
