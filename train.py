# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10.py
import math
import os
import time
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.python.platform import gfile

from hyperparams import OPTIMIZER, FLAGS, OPTIMIZER_ARGS
from util import format_time_hhmmss


def train(dataset, net):
    if not gfile.Exists(FLAGS.checkpoint_path):
        gfile.MkDir(FLAGS.checkpoint_path)
    if not gfile.Exists(FLAGS.summary_path):
        gfile.MkDir(FLAGS.summary_path)

    now = dt.now()
    exp_dirname = FLAGS.experiment_name + ('_%s' % now.strftime('%Y-%m-%d_%H-%M-%S'))
    summary_path = os.path.join(FLAGS.summary_path, exp_dirname)
    checkpoint_path = os.path.join(FLAGS.checkpoint_path, exp_dirname)
    gfile.MkDir(summary_path)
    gfile.MkDir(checkpoint_path)

    dataset.pre_graph()

    # with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    dataset.preliminary()

    global_step = tf.Variable(0, trainable=False)

    # input and training procedure
    images, true_labels = dataset.training_inputs()
    predictions = net(images, dataset.num_classes)
    loss_op = loss(predictions, true_labels)
    train_err_op = training_error(predictions, true_labels)
    train_op = training_op(loss_op, train_err_op, global_step)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

    coord = tf.train.Coordinator()

    # perform initial ops
    sess.run(init_op)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.train.SummaryWriter(summary_path, sess.graph_def)

    # the training loop
    print '%s - Started training.' % dt.now()
    overall_start_time = time.time()
    while not coord.should_stop():

        # measure computation time of the costly operations
        start_time = time.time()
        _, loss_value, train_err_value = sess.run([train_op, loss_op, train_err_op])
        duration = time.time() - start_time

        assert not math.isnan(loss_value), 'Model diverged with loss = NaN.'

        step = sess.run(global_step)
        # print timing information
        # TODO make module bigger
        if step % FLAGS.log_interval == 0:
            examples_per_step = FLAGS.batch_size
            examples_per_sec = examples_per_step / duration
            sec_per_batch = float(duration)
            print '%s - step %d, loss = %.2f, training error = %.2f%% (%.1f examples/sec; %.3f sec/batch)' % (
                dt.now(), step, loss_value, train_err_value * 100, examples_per_sec, sec_per_batch)

        # add summaries
        # TODO make module bigger
        if step % FLAGS.summary_interval == 0:
            summary = sess.run(summary_op)
            summary_writer.add_summary(summary, step)

        # periodically save progress
        if step % FLAGS.checkpoint_interval == 0 or step == FLAGS.training_epochs:
            saver.save(sess, os.path.join(checkpoint_path, 'model.ckpt'), global_step=step)

        # reached epoch limit - done with training
        if step == FLAGS.training_epochs:
            coord.request_stop()
            overall_duration = time.time() - overall_start_time
            print '%s - Epoch limit reached. Done training in %s.' % (
                dt.now(), format_time_hhmmss(overall_duration))
            break

    coord.join(threads)
    sess.close()


def _activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def loss(predictions, true_labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predictions, true_labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + '_raw', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def training_error(predictions, true_labels):
    softmaxed = tf.nn.softmax(predictions)
    correct_prediction = tf.equal(tf.argmax(softmaxed, 1), tf.argmax(true_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    training_err = 1 - accuracy
    return training_err


def _add_train_err_summaries(training_err):
    train_err_avg = tf.train.ExponentialMovingAverage(0.9, name='train_err_avg')
    train_err_avg_op = train_err_avg.apply([training_err])
    tf.scalar_summary('training_error_raw', training_err)
    tf.scalar_summary('training_error_averaged', train_err_avg.average(training_err))

    return train_err_avg_op


def training_op(total_loss, train_err, global_step):
    # TODO add correct learning rate decay
    lr = tf.train.exponential_decay(0.01, global_step, 100, 0.1, staircase=True)
    tf.scalar_summary('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)
    train_err_avg_op = _add_train_err_summaries(train_err)

    with tf.control_dependencies([loss_averages_op, train_err_avg_op]):
        optimizer = OPTIMIZER(lr, **OPTIMIZER_ARGS)
        grads = optimizer.compute_gradients(total_loss)

    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
