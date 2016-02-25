# some learning rate decay strategies
import tensorflow as tf


def decay_at_fixed_steps_default(lr, global_step):
    return decay_at_fixed_steps(lr, global_step, [32000, 48000], 0.1)


def decay_at_fixed_steps(lr, global_step, thresholds, decay_factor):
    thresholds_tensor = tf.constant(thresholds, dtype=tf.int32)
    step_replicated = tf.tile(tf.expand_dims(global_step, 0), [thresholds_tensor.get_shape().as_list()[0]])
    new_lr = tf.cond(tf.reduce_any(tf.equal(step_replicated, thresholds_tensor)),
                     lambda: lr * decay_factor,
                     lambda: lr)
    return new_lr


def raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err):
    return raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, [32000, 48000], 0.1, train_err, 0.8, 0.1)


def raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, thresholds, decay_factor, train_err, err_thresh,
                                                 raise_to):
    err_thresh_tensor = tf.constant(err_thresh, dtype=tf.float32)

    new_lr = tf.cond(tf.less(train_err, err_thresh_tensor),
                     lambda: _raise_and_continue_normal_decay(lr, global_step, thresholds, decay_factor, raise_to),
                     lambda: lr)
    return new_lr


def _raise_and_continue_normal_decay(lr, global_step, thresholds, decay_factor, raise_to):
    lr = lr.assign(raise_to)
    return decay_at_fixed_steps(lr, global_step, thresholds, decay_factor)
