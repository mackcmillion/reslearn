# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_eval.py
import time

import tensorflow as tf
from datetime import datetime as dt

from config import FLAGS


def validate(dataset, model, summary_path, checkpoint_path):

    # input and validation procedure
    images, true_labels = dataset.validation_inputs()
    predictions = model.inference(images, dataset.num_classes)
    top_k_op = _top_k_10crop(predictions, true_labels)

    saver = tf.train.Saver(tf.trainable_variables())

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(summary_path, tf.get_default_graph().as_graph_def())

    finished = False
    while not finished:
        finished = _eval_once(saver, checkpoint_path, summary_writer, top_k_op, summary_op)
        if FLAGS.run_once:
            break
        time.sleep(FLAGS.val_interval_secs)


def _eval_once(saver, checkpoint_path, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print '%s - No new checkpoint file found.' % dt.now()
            return False

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            true_count = 0
            step = 0
            while step < FLAGS.max_num_examples and not coord.should_stop():
                prediction = sess.run([top_k_op])
                true_count += prediction
                step += 1

            accuracy = true_count / step
            validation_error = 1 - accuracy
            print '%s - validation error = %.3f' % (dt.now(), validation_error)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add('validation_error_raw', simple_value=validation_error)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def _top_k_10crop(predictions, true_labels):
    pred_mean = tf.reduce_mean(predictions, reduction_indices=0)
    return tf.nn.in_top_k(pred_mean, true_labels, FLAGS.top_k)