import threading

import tensorflow as tf
import os

from tensorflow.python.platform import gfile

from hyperparams import FLAGS
from preprocess import preprocess

# TODO adjust these constants
MIN_NUM_EXAMPLES_IN_QUEUE = 10
NUM_PRODUCING_THREADS = 1
NUM_CONSUMING_THREADS = 1

WNID_LID_MAP = None


def inputs():
    filenames = _build_filename_list()

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=FLAGS.training_epochs, shuffle=True)

    example_list = [_read_and_preprocess_image(filename_queue) for _ in xrange(NUM_CONSUMING_THREADS)]

    image_batch, label_batch = tf.train.shuffle_batch_join(
            example_list,
            batch_size=FLAGS.batch_size,
            capacity=MIN_NUM_EXAMPLES_IN_QUEUE + (NUM_CONSUMING_THREADS + 2) * FLAGS.batch_size,
            min_after_dequeue=MIN_NUM_EXAMPLES_IN_QUEUE
    )

    return image_batch, label_batch
    # return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])


def _build_filename_list():
    image_files = []
    for dirpath, _, filenames in os.walk(FLAGS.train_dir):
        image_files += [os.path.join(dirpath, filename) for filename in filenames]
    return image_files


def _read_and_preprocess_image(filename_queue):
    reader = tf.WholeFileReader()
    filename, image = reader.read(filename_queue)

    # prepare image
    image = tf.image.decode_jpeg(image, channels=3)
    feature_img = tf.cast(image, tf.float32)
    processed_img = preprocess(feature_img)

    # TODO implement label loading
    # prepare labels
    label = _get_label_id_for_wnid(filename)
    print label
    lbls = tf.zeros([FLAGS.num_classes], dtype=tf.int32)

    return processed_img, lbls


def _get_label_id_for_wnid(filename):
    # first access, map not yet loaded
    if not WNID_LID_MAP:
        _load_wnid_lid_map()

    wnid = filename.split('_')[0]
    if wnid not in WNID_LID_MAP:
        raise KeyError('Unknown WNID ' + wnid)
    return WNID_LID_MAP[wnid][0]


def _load_wnid_lid_map():
    if not gfile.Exists(FLAGS.wnid_lid_path):
        raise ValueError('WNID_LID file not found.')
    WNID_LID_MAP = dict()

    f = open(FLAGS.wnid_lid_path, 'r')
    for line in f:
        contents = line.split(' ')
        WNID_LID_MAP[contents[0]] = (int(contents[1]), contents[2])


def compute_overall_mean_stddev(overwrite=False, num_threads=1):

    if gfile.Exists(FLAGS.mean_stddev_path):
        print 'Mean/stddev file already exists.'
        if overwrite:
            print 'Overwriting file...'
            gfile.Remove(FLAGS.mean_stddev_path)
        else:
            print 'Nothing to do here.'
            return
        print

    print 'Building filename list...'
    filenames = _build_filename_list()

    # mean computation
    mean = tf.Variable([0.0, 0.0, 0.0])
    total = tf.Variable(0.0)

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.cast(image, tf.float32)

    sum = tf.reduce_sum(image, [0, 1])
    num = tf.mul(tf.shape(image)[0], tf.shape(image)[1])
    num = tf.cast(num, tf.float32)

    queue = tf.FIFOQueue(1000, dtypes=[tf.float32, tf.float32], shapes=[[3], []])
    enqueue_op = queue.enqueue([sum, num])

    img_sum, img_num = queue.dequeue()
    mean_op = tf.add(mean, img_sum)
    total_op = tf.add(total, img_num)

    qr = tf.train.QueueRunner(queue, [enqueue_op] * num_threads)
    tf.train.add_queue_runner(qr)

    init_op = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_no = 0
    print 'Computing mean...'
    try:
        while not coord.should_stop():
            image_no += 1
            print image_no
            mean, total = sess.run([mean_op, total_op])

    except tf.errors.OutOfRangeError:
        print 'All images processed.'
    finally:
        coord.request_stop()

    coord.join(threads)

    total_3channel = tf.pack([total, total, total])
    mean = tf.div(mean, total_3channel)
    mean = sess.run(mean)
    print mean

    sess.close()

    # stddev computation
    variance = tf.zeros([3], dtype=tf.float32)
    total = tf.zeros([], dtype=tf.float32)

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.cast(image, tf.float32)

    num = tf.mul(tf.shape(image)[0], tf.shape(image)[1])

    mean_tiled = tf.tile(mean, tf.expand_dims(num, 0))
    mean_tiled = tf.reshape(mean_tiled, tf.pack([tf.shape(image)[0], tf.shape(image)[1], 3]))

    remainders = tf.sub(image, mean_tiled)
    squares = tf.square(remainders)
    sum_of_squares = tf.reduce_sum(squares, [0, 1])

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_no = 0
    print 'Computing standard deviation.'
    try:
        while not coord.should_stop():
            image_no += 1
            if image_no % 1000 == 0:
                print 'Image number %i' % image_no
            sum_sq, img_num = sess.run([sum_of_squares, num])
            variance = tf.add(variance, sum_sq)
            total = tf.add(total, img_num)

    except tf.errors.OutOfRangeError:
        print 'All images processed.'
    finally:
        coord.request_stop()

    coord.join(threads)

    total = total - tf.constant(1.0)
    total_3channel = tf.pack([total, total, total])
    variance = tf.div(variance, total_3channel)
    stddev = tf.sqrt(variance)
    stddev = sess.run(stddev)

    sess.close()

    f = open(FLAGS.mean_stddev_path, 'w')
    f.write('%s\n%s\n' % (str(mean), str(stddev)))
    f.close()


def _parallel_sum_for_mean(coord, sess, sum, num, means, totals, index):
    mean = tf.zeros([3])
    total = tf.zeros([])
    try:
        while not coord.should_stop():
            img_sum, img_num = sess.run([sum, num])
            mean = tf.add(mean, img_sum)
            total = tf.add(total, img_num)

    except tf.errors.OutOfRangeError:
        print 'All images processed.'
    finally:
        coord.request_stop()

    means[index] = sess.run(mean)
    totals[index] = sess.run(total)


if __name__ == '__main__':
    compute_overall_mean_stddev(overwrite=True, num_threads=4)
