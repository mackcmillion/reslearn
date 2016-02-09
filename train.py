import tensorflow as tf
from input import inputs
from hyperparams import NET, OPTIMIZER


def train():

    training = train_op()

    init_op = tf.initialize_all_variables()

    sess = tf.Session()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('summaries/resnet_34_test', sess.graph_def)

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0

    try:
        while not coord.should_stop():
            step += 1
            print 'Training step %i' % step
            sess.run(training)

    except tf.errors.OutOfRangeError:
        print 'Done training - epoch limit reached.'
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def train_op():

    images, true_labels = inputs()

    predictions = NET(images)

    true_labels = tf.cast(true_labels, tf.float32)

    loss = tf.nn.softmax_cross_entropy_with_logits(predictions, true_labels)

    return OPTIMIZER.minimize(loss)
