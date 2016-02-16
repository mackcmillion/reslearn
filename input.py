import os

import tensorflow as tf
from tensorflow.python.platform import gfile

import util
from hyperparams import FLAGS
from preprocess import preprocess_for_training, preprocess_for_validation

# TODO adjust these constants
MIN_NUM_EXAMPLES_IN_QUEUE = 10
NUM_CONSUMING_THREADS = 1


def training_inputs():
    filepaths, labels = _load_training_labelmap()
    filepaths = tf.constant(filepaths)
    labels = tf.constant(labels, dtype=tf.int32)

    filename_queue = tf.RandomShuffleQueue(len(filepaths), MIN_NUM_EXAMPLES_IN_QUEUE, [tf.string, tf.int32],
                                           name='filename_queue')
    enqueue_op = filename_queue.enqueue_many([filepaths, labels])
    qr = tf.train.QueueRunner(filename_queue, [enqueue_op])
    tf.train.add_queue_runner(qr)

    example_list = [_read_and_preprocess_image_for_training(filename_queue) for _ in xrange(NUM_CONSUMING_THREADS)]

    image_batch, filename_batch = tf.train.shuffle_batch_join(
            example_list,
            batch_size=FLAGS.batch_size,
            capacity=MIN_NUM_EXAMPLES_IN_QUEUE + (NUM_CONSUMING_THREADS + 2) * FLAGS.batch_size,
            min_after_dequeue=MIN_NUM_EXAMPLES_IN_QUEUE,
            shapes=[[224, 224, 3], []]
    )

    return image_batch, util.encode_one_hot(filename_batch, FLAGS.num_classes)


def _load_training_labelmap():
    if not gfile.Exists(FLAGS.training_set):
        raise ValueError('Training label map file not found.')

    filepaths = []
    labels = []

    with open(FLAGS.training_set) as f:
        for line in f:
            fp, lbl = line.split(',')
            filepaths.append(fp)
            labels.append(int(lbl[:-1]))

    return filepaths, labels


def _read_and_preprocess_image_for_training(filename_queue):
    filename, label = filename_queue.dequeue()

    # we want to reuse the filename-label-pair for later training steps
    filename_queue.enqueue([filename, label])

    image = tf.read_file(filename)

    # prepare image
    image = tf.image.decode_jpeg(image, channels=3)
    feature_img = tf.cast(image, tf.float32)
    processed_img = preprocess_for_training(feature_img)

    return [processed_img, label]


def validation_inputs():
    filepaths, labels = _load_validation_labelmap()
    filepaths = tf.constant(filepaths)
    labels = tf.constant(labels, dtype=tf.int32)

    filename_queue = tf.FIFOQueue(len(filepaths), [tf.string, tf.int32], name='filename_queue')
    enqueue_op = filename_queue.enqueue_many([filepaths, labels])
    qr = tf.train.QueueRunner(filename_queue, [enqueue_op])
    tf.train.add_queue_runner(qr)

    example_queue = tf.FIFOQueue(len(filepaths), [tf.float32, tf.int32], name='example_queue')
    enqueue_op_ex = example_queue.enqueue(_read_and_preprocess_image_for_validation(filename_queue))
    qr_ex = tf.train.QueueRunner(example_queue, [enqueue_op_ex] * NUM_CONSUMING_THREADS)
    tf.train.add_queue_runner(qr_ex)

    image_10crop, label = example_queue.dequeue()
    # don't one-hot-encode label here
    return image_10crop, label


def _load_validation_labelmap():
    if not gfile.Exists(FLAGS.validation_set):
        raise ValueError('Validation label map not found.')

    filepaths = []
    labels = []

    with open(FLAGS.validation_set) as f:
        for line_no, line in enumerate(f):
            fp = os.path.join(FLAGS.validation_dir, 'ILSVRC2012_val_%08d.JPEG' % line_no)
            filepaths.append(fp)
            line_no += 1
            labels.append(int(line))

    return filepaths, labels


def _read_and_preprocess_image_for_validation(filename_queue):
    filename, label = filename_queue.dequeue()

    image = tf.read_file(filename)

    # prepare image
    image = tf.image.decode_jpeg(image, channels=3)
    feature_img = tf.cast(image, tf.float32)
    processed_img = preprocess_for_validation(feature_img)

    return [processed_img, label]
