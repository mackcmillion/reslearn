# some learning rate decay strategies
from config import FLAGS

_RAISED = False


def update_lr(sess, lr, global_step, train_err_avg, overall_train_err):
    if FLAGS.learning_rate_decay_strategy == 0:
        return decay_at_fixed_steps_default(lr, global_step)
    elif FLAGS.learning_rate_decay_strategy == 1:
        train_err = sess.run(train_err_avg)
        return raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err)
    elif FLAGS.learning_rate_decay_strategy == 2:
        return dynamic_decay(lr, global_step, 0.1, overall_train_err)
    else:
        raise ValueError('Unknown learning rate decay strategy.')


def decay_at_fixed_steps_default(lr, global_step):
    return decay_at_fixed_steps(lr, global_step, [32000, 48000], 0.1)


def decay_at_fixed_steps(lr, global_step, thresholds, decay_factor):
    if any(global_step == t for t in thresholds):
        return lr * decay_factor
    return lr


def raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err):
    return raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, [32000, 48000], 0.1, train_err, 0.8, 0.1)


def raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, thresholds, decay_factor, train_err, err_thresh,
                                                 raise_to):
    global _RAISED
    if _RAISED:
        return decay_at_fixed_steps(lr, global_step, thresholds, decay_factor)
    elif train_err < err_thresh:
        _RAISED = True
        return raise_to
    return lr


PREV_TRAIN_ERR = 1.0
# maximum of decays should be 2
NUM_DECAYS = 0


def dynamic_decay(lr, global_step, decay_factor, train_err):
    global PREV_TRAIN_ERR, NUM_DECAYS
    if NUM_DECAYS < 2 and train_err < PREV_TRAIN_ERR:
        new_lr = lr
    else:
        new_lr = decay_factor * lr
        NUM_DECAYS += 1
    PREV_TRAIN_ERR = train_err
    return new_lr


def dynamic_decay_default(lr, global_step, train_err):
    return dynamic_decay(lr, global_step, 0.1, train_err)
