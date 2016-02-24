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
