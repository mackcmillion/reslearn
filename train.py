import tensorflow as tf

import util
from hyperparams import NET, OPTIMIZER, FLAGS
from input import inputs


def train():

    # images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 224, 224, 3])
    # true_labels = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size])

    training = train_step()
    # ins = inputs()

    init_op = tf.initialize_all_variables()

    sess = tf.Session()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('summaries/resnet_34_test', sess.graph_def)

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    # try:
    #     while not coord.should_stop():
    #         print step
    #         _training_loop(sess, training)
    #         step += 1
    #
    # except tf.errors.OutOfRangeError:
    #     print 'Done training - epoch limit reached.'
    # finally:
    #     coord.request_stop()
    while True:
        print step
        _training_loop(sess, training)
        step += 1

    coord.join(threads)
    sess.close()


def _training_loop(sess, training):

    # image_batch, filename_batch = sess.run(ins)
    # label_batch = map(_get_label_id_for_wnid, filename_batch)
    sess.run(training)


def train_step():

    images, true_labels = inputs()

    predictions = NET(images)

    # true_labels = util.encode_one_hot(label_batch, FLAGS.num_classes)

    loss = tf.nn.softmax_cross_entropy_with_logits(predictions, true_labels)

    return OPTIMIZER.minimize(loss)
