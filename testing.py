import tensorflow as tf

from hyperparams import NET
from input import inputs


def test():

    k = 1

    sess = tf.Session()

    testing = test_step(k)
    init_op = tf.initialize_all_variables()

    # merged = tf.merge_all_summaries()
    # writer = tf.train.SummaryWriter('summaries/resnet_34_test', sess.graph_def)

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    correct = 0
    total = 0
    try:
        while not coord.should_stop():
            print step
            _testing_loop(sess, testing, correct, total)
            step += 1

    except tf.errors.OutOfRangeError:
        print 'Done training - epoch limit reached.'
    finally:
        coord.request_stop()

    coord.join(threads)

    print 'Top %i error: %f' % (k, (correct * 1.0) / total)

    sess.close()


def _testing_loop(sess, testing, correct, total):
    in_top_k = sess.run(testing)
    if in_top_k:
        correct += 1
    total += 1
    return correct, total


def test_step(k):

    # TODO write custom testing input function
    # FIXME true_label needs to be the label index
    crops, true_label = inputs()

    predictions = NET(crops)

    batch_size = tf.shape(predictions)[0]
    pred_sum = tf.reduce_sum(predictions, reduction_indices=0)
    average = pred_sum / batch_size

    return tf.nn.in_top_k(average, true_label, k)
