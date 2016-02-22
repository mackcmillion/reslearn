# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_eval.py
import time

import tensorflow as tf
from datetime import datetime as dt

from config import FLAGS


def validate(dataset, model, summary_path, checkpoint_path):

    with tf.Graph().as_default():
        # input and validation procedure
        images, true_labels = dataset.validation_inputs()
        predictions = model.inference(images, dataset.num_classes)
        top_k_op = _top_k_10crop(predictions, true_labels)

        saver = tf.train.Saver(tf.trainable_variables())

        val_err = tf.placeholder(tf.float32, shape=[], name='val_err')
        tf.scalar_summary('validation_error_raw', val_err)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(summary_path, tf.get_default_graph().as_graph_def())

        finished = False
        while not finished:
            finished = _eval_once(saver, checkpoint_path, summary_writer, top_k_op, summary_op, val_err)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.val_interval_secs)


def _eval_once(saver, checkpoint_path, summary_writer, top_k_op, summary_op, val_err):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
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
            true_count = 0
            step = 0
            while step < FLAGS.max_num_examples and not coord.should_stop():
                # TODO this is so damn slow since we're not processing batches
                prediction = sess.run([top_k_op])
                print prediction
                if prediction[0]:
                    true_count += 1
                step += 1

            accuracy = true_count / step
            validation_error = 1 - accuracy
            print '%s - validation error = %.3f' % (dt.now(), validation_error)

            summary = sess.run(summary_op, feed_dict={val_err: validation_error})
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        return global_step == FLAGS.training_epochs


def _top_k_10crop(predictions, true_labels):
    pred_mean = tf.reduce_mean(predictions, reduction_indices=0)
    pred_mean = tf.expand_dims(pred_mean, 0)
    true_labels = tf.expand_dims(true_labels, 0)
    return tf.nn.in_top_k(pred_mean, true_labels, FLAGS.top_k)
