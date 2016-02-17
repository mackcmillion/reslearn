import math
import os
import time
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.python.platform import gfile

from hyperparams import OPTIMIZER, FLAGS
from input import training_inputs
from util import format_time_hhmmss

from scripts.labelmap import create_label_map_file


def train(net):

    preliminary_computations()

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # input and training procedure
        images, true_labels = training_inputs()
        predictions = net(images)
        loss_per_example = tf.nn.softmax_cross_entropy_with_logits(predictions, true_labels)
        loss = tf.reduce_mean(loss_per_example, name='cross_entropy')
        # the minimize operation also increments the global step
        train_op = OPTIMIZER.minimize(loss, global_step=global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        coord = tf.train.Coordinator()

        # perform initial ops
        sess.run(init_op)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.train.SummaryWriter(FLAGS.summary_path, sess.graph_def)

        # the training loop
        print '%s - Started training.' % dt.now()
        overall_start_time = time.time()
        while not coord.should_stop():

            # measure computation time of the costly operations
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not math.isnan(loss_value), 'Model diverged with loss = NaN.'

            step = sess.run(global_step)
            # print timing information
            if step % 1 == 0:
                examples_per_step = FLAGS.batch_size
                examples_per_sec = examples_per_step / duration
                sec_per_batch = float(duration)
                print '%s - step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
                    dt.now(), step, loss_value, examples_per_sec, sec_per_batch)

            # add summaries
            if step % 100 == 0:
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary, step)

            # periodically save progress
            if step % 1000 == 0 or step == FLAGS.training_epochs:
                saver.save(sess, os.path.join(FLAGS.checkpoint_path, 'model.ckpt'), global_step=step)

            # reached epoch limit - done with training
            if step == FLAGS.training_epochs:
                coord.request_stop()
                overall_duration = time.time() - overall_start_time
                print '%s - Epoch limit reached. Done training in %s.' % (
                    dt.now(), format_time_hhmmss(overall_duration))
                break

        coord.join(threads)
        sess.close()


def preliminary_computations():
    if not gfile.Exists(FLAGS.checkpoint_path):
        gfile.MkDir(FLAGS.checkpoint_path)

    create_label_map_file(overwrite=True)
