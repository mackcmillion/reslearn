import tensorflow as tf

from hyperparams import NET, OPTIMIZER
from input import training_inputs


def train():

    sess = tf.Session()

    training = train_step()
    init_op = tf.initialize_all_variables()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('summaries/resnet_34_test', sess.graph_def)

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    try:
        while not coord.should_stop():
            print step
            _training_loop(sess, training, merged, writer, step)
            step += 1
            # TODO remove
            if step == 3:
                coord.request_stop()

    except tf.errors.OutOfRangeError:
        print 'Done training - epoch limit reached.'
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def _training_loop(sess, training, merge, writer, step):

    sess.run(training)
    summ = sess.run(merge)
    writer.add_summary(summ, step)


def train_step():

    images, true_labels = training_inputs()

    tf.image_summary('test_image_summary', images)

    predictions = NET(images)

    loss = tf.nn.softmax_cross_entropy_with_logits(predictions, true_labels)

    return OPTIMIZER.minimize(loss)
