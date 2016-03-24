# some learning rate decay strategies
from config import FLAGS

_RAISED = False


def update_lr(sess, lr, global_step, train_err_avg):
    if FLAGS.learning_rate_decay_strategy == 0:
        return decay_at_fixed_steps_default(lr, global_step)
    elif FLAGS.learning_rate_decay_strategy == 1:
        train_err = sess.run([train_err_avg])
        return raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err)
    else:
        raise ValueError('Unknown learning rate decay strategy.')


def decay_at_fixed_steps_default(lr, global_step):
    return decay_at_fixed_steps(lr, global_step, [52000, 92000], 0.1)


def decay_at_fixed_steps(lr, global_step, thresholds, decay_factor):
    if any(global_step == t for t in thresholds):
        return lr * decay_factor
    return lr


def raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err):
    return raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, [52000, 92000], 0.1, train_err, 0.76, 0.1)


def raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, thresholds, decay_factor, train_err, err_thresh,
                                                 raise_to):
    if _RAISED:
        return decay_at_fixed_steps(lr, global_step, thresholds, decay_factor)
    elif train_err < err_thresh:
        return raise_to
    return lr
