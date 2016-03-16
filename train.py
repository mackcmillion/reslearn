# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10.py
import math
import os
import time
from datetime import datetime as dt

import tensorflow as tf

import learningrate
import util
from config import OPTIMIZER, FLAGS, OPTIMIZER_ARGS
from util import format_time_hhmmss


def train(dataset, model, summary_path, checkpoint_path):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    dataset.preliminary()

    # input and training procedure
    images, true_labels = dataset.training_inputs()
    predictions = model.inference(images, dataset.num_classes, True)
    loss_op = loss(predictions, true_labels)
    train_err = tf.Variable(1.0, trainable=False)
    train_err_assign = training_error(predictions, true_labels, train_err)

    global_step = 0
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

    # optionally resume from existing checkpoint
    ckpt = None
    if FLAGS.resume:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = util.extract_global_step(ckpt.model_checkpoint_path)
            print '%s - Resuming training from global step %i.' % (dt.now(), global_step)
        else:
            print '%s - No checkpoint found. Starting fresh run of %s.' % (dt.now(), FLAGS.experiment_name)

    global_step = tf.Variable(global_step, trainable=False)
    train_op = training_op(loss_op, train_err, train_err_assign, global_step)

    # restore here to make sure all variables are defined before assigning them
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

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
        _, loss_value = sess.run([train_op, loss_op])
        duration = time.time() - start_time

        assert not math.isnan(loss_value), '%s - Model diverged with loss = NaN.' % dt.now()

        step = sess.run(global_step)
        # print timing information
        if step % FLAGS.log_interval == 0:
            examples_per_step = FLAGS.batch_size
            examples_per_sec = examples_per_step / duration
            sec_per_batch = float(duration)
            print '%s - step %d, loss = %.2f, training error = %.2f%% (%.1f examples/sec; %.3f sec/batch)' % (
                dt.now(), step, loss_value, sess.run(train_err) * 100, examples_per_sec, sec_per_batch)

        # add summaries
        if step % FLAGS.summary_interval == 0:
            summary = sess.run(summary_op)
            summary_writer.add_summary(summary, step)

        # periodically save progress
        if step % FLAGS.checkpoint_interval == 0 or step == FLAGS.training_steps:
            saver.save(sess, os.path.join(checkpoint_path, model.name + '.ckpt'), global_step=step)

        # reached step limit - done with training
        if step == FLAGS.training_steps:
            coord.request_stop()
            overall_duration = time.time() - overall_start_time
            print '%s - Step limit reached. Done training in %s.' % (
                dt.now(), format_time_hhmmss(overall_duration))
            break

    coord.join(threads)
    sess.close()


def _activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def loss(predictions, true_labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, true_labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + '_raw', l)
        tf.scalar_summary(l.op.name + '_averaged', loss_averages.average(l))

    return loss_averages_op


def training_error(predictions, true_labels, train_err):
    softmaxed = tf.nn.softmax(predictions)
    correct_prediction = tf.equal(tf.argmax(softmaxed, 1), tf.argmax(true_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_err_op = 1 - accuracy

    train_err_assign = train_err.assign(train_err_op)
    return train_err_assign


def _add_train_err_summaries(train_err):
    train_err_avg_obj = tf.train.ExponentialMovingAverage(0.9, name='train_err_avg')
    train_err_avg_op = train_err_avg_obj.apply([train_err])

    tf.scalar_summary('training_error_raw', train_err)
    averaged = train_err_avg_obj.average(train_err)
    tf.scalar_summary('training_error_averaged', averaged)

    return train_err_avg_op, averaged


def training_op(total_loss, train_err, train_err_assign, global_step):
    loss_averages_op = _add_loss_summaries(total_loss)
    train_err_avg_op, train_err_avg = _add_train_err_summaries(train_err)

    lr = tf.Variable(FLAGS.initial_learning_rate, name='learning_rate', trainable=False)
    lr_decay_op = lr.assign(
            [
                learningrate.decay_at_fixed_steps_default(lr, global_step),
                learningrate.raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err_avg)
            ][FLAGS.learning_rate_decay_strategy]
    )
    tf.scalar_summary('learning_rate_summary', lr)

    with tf.control_dependencies([train_err_avg_op, train_err_assign]):
        with tf.control_dependencies([loss_averages_op, lr_decay_op]):
            optimizer = OPTIMIZER(lr, **OPTIMIZER_ARGS)
            grads = optimizer.compute_gradients(total_loss)

    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # this is not used in the paper
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op
                                  ]):
        train_op = tf.no_op(name='train')

    return train_op
