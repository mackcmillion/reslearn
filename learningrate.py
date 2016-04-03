# some learning rate decay strategies
from config import FLAGS

_RAISED = False


def update_lr(sess, lr, global_step, train_err_avg):
    if FLAGS.learning_rate_decay_strategy == 0:
        return decay_at_fixed_steps_default(lr, global_step)
    elif FLAGS.learning_rate_decay_strategy == 1:
        train_err = sess.run(train_err_avg)
        return raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err)
    else:
        raise ValueError('Unknown learning rate decay strategy.')


def decay_at_fixed_steps_default(lr, global_step):
    return decay_at_fixed_steps(lr, global_step, [52000, 72000], 0.1)


def decay_at_fixed_steps(lr, global_step, thresholds, decay_factor):
    if any(global_step == t for t in thresholds):
        return lr * decay_factor
    return lr


def raise_at_train_err_then_decay_at_fixed_steps_default(lr, global_step, train_err):
    return raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, [52000, 72000], 0.1, train_err, 0.8, 0.1)


def raise_at_train_err_then_decay_at_fixed_steps(lr, global_step, thresholds, decay_factor, train_err, err_thresh,
                                                 raise_to):
    global _RAISED
    if _RAISED:
        return decay_at_fixed_steps(lr, global_step, thresholds, decay_factor)
    elif train_err < err_thresh:
        _RAISED = True
        return raise_to
    return lr


AVG_ACCUM = []
DECAY_ACCUM = [1.0, -1.0, 1.0, -1.0, -1.0]
SAFETY_TIMER = 10000
OLD_GLOBAL_STEP = 0


def dynamic_decay(lr, global_step, decay_factor, train_err):
    global AVG_ACCUM, DECAY_ACCUM, SAFETY_TIMER, OLD_GLOBAL_STEP

    if global_step % 5000 == 0:
        avg = sum(AVG_ACCUM) / len(AVG_ACCUM)
        DECAY_ACCUM.append(avg)
        DECAY_ACCUM.pop(0)

        AVG_ACCUM = []

        if max(DECAY_ACCUM) - min(DECAY_ACCUM) <= 0.01 and SAFETY_TIMER <= 0:
            SAFETY_TIMER = 10000
            OLD_GLOBAL_STEP = global_step
            return lr * decay_factor

    AVG_ACCUM.append(train_err)
    SAFETY_TIMER -= global_step - OLD_GLOBAL_STEP
    OLD_GLOBAL_STEP = global_step
    return lr


def dynamic_decay_default(lr, global_step, train_err):
    return dynamic_decay(lr, global_step, 0.1, train_err)
