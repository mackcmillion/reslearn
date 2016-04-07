# most of this code is taken from
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10.py
import math
import numpy
import os
import time
from datetime import datetime as dt

import tensorflow as tf

import learningrate
import util
from config import FLAGS
from util import format_time_hhmmss


def train(dataset, model, summary_path, checkpoint_path):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    dataset.preliminary()

    # input and training procedure
    images, true_labels = dataset.training_inputs()
    predictions = model.inference(images, dataset.num_classes, True)
    loss_op = loss(dataset, predictions, true_labels)
    train_err = tf.Variable(1.0, trainable=False)
    train_err_assign, train_err_name = training_error(predictions, true_labels, dataset, train_err)

    eval_op = dataset.eval_op(predictions, true_labels)
    overall_train_err_op = dataset.test_error

    lr = FLAGS.initial_learning_rate
    learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

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
    train_op, train_err_avg = training_op(loss_op, train_err, train_err_assign, learning_rate, global_step)

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
        _, loss_value = sess.run([train_op, loss_op], feed_dict={learning_rate: lr})
        duration = time.time() - start_time

        assert not math.isnan(loss_value), '%s - Model diverged with loss = NaN.' % dt.now()

        step = sess.run(global_step)
        # print timing information
        if step % FLAGS.log_interval == 0:
            examples_per_step = FLAGS.batch_size
            examples_per_sec = examples_per_step / duration
            sec_per_batch = float(duration)
            print '%s - step %d, loss = %.2f, %s = %.2f%%, lr = %.3f (%.1f examples/sec; %.3f sec/batch)' \
                  % (dt.now(), step, loss_value, train_err_name, sess.run(train_err) * 100, lr, examples_per_sec,
                     sec_per_batch)

        # add summaries
        if step % FLAGS.summary_interval == 0:
            summary = sess.run(summary_op, feed_dict={learning_rate: lr})
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

        if step % FLAGS.lr_interval == 0:
            if FLAGS.learning_rate_decay_strategy == 2:
                overall_train_err = _compute_overall_training_error(sess, dataset, predictions, true_labels, eval_op,
                                                                    overall_train_err_op)
            else:
                # value is not needed for other LR schedules
                overall_train_err = 0
            # update learning rate if necessary
            lr = learningrate.update_lr(sess, lr, step, train_err_avg, overall_train_err)

    coord.join(threads)
    sess.close()


def _activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def loss(dataset, predictions, true_labels):
    loss_ = dataset.loss_fn(predictions, true_labels)
    loss_mean = tf.reduce_mean(loss_, name='loss')
    tf.add_to_collection('losses', loss_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + '_raw', l)
        tf.scalar_summary(l.op.name + '_averaged', loss_averages.average(l))

    return loss_averages_op


def training_error(predictions, true_labels, dataset, train_err):
    train_err_op, train_err_name = dataset.training_error(predictions, true_labels)

    train_err_assign = train_err.assign(train_err_op)
    return train_err_assign, train_err_name


def _add_train_err_summaries(train_err):
    train_err_avg_obj = tf.train.ExponentialMovingAverage(0.9, name='train_err_avg')
    train_err_avg_op = train_err_avg_obj.apply([train_err])

    tf.scalar_summary('training_error_raw', train_err)
    averaged = train_err_avg_obj.average(train_err)
    tf.scalar_summary('training_error_averaged', averaged)

    return train_err_avg_op, averaged


def training_op(total_loss, train_err, train_err_assign, learning_rate, global_step):
    loss_averages_op = _add_loss_summaries(total_loss)
    train_err_avg_op, train_err_avg = _add_train_err_summaries(train_err)

    tf.scalar_summary('learning_rate_summary', learning_rate)

    with tf.control_dependencies([train_err_avg_op, train_err_assign]):
        with tf.control_dependencies([loss_averages_op]):
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)
            grads = optimizer.compute_gradients(total_loss)

    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # this is not used in the paper
    # variable_averages = tf.train.ExponentialMovingAverage(FLAGS.variable_average_decay, global_step)
    # variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op  # , variable_averages_op
                                  ]):
        train_op = tf.no_op(name='train')

    return train_op, train_err_avg


def _compute_overall_training_error(sess, coord, global_step, dataset, eval_op, overall_train_err_op):
    num_iter = int(math.ceil((1.0 * dataset.num_training_images) / FLAGS.batch_size))
    accumulated = 0.0
    total_sample_count = num_iter * FLAGS.batch_size
    step = 0
    while step < num_iter and not coord.should_stop():
        predictions = sess.run(eval_op)
        accumulated += numpy.sum(predictions)
        step += 1

    overall_train_error, overall_train_error_name = overall_train_err_op(accumulated, total_sample_count)
    print '%s - step %d, %s = %.2f%%' % (dt.now(), global_step, overall_train_error_name, overall_train_error * 100)

    return overall_train_error
