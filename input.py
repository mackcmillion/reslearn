import tensorflow as tf
from tensorflow.python.platform import gfile

import util
from hyperparams import FLAGS
from preprocess import preprocess

from scripts.labelmap import build_filename_list

# TODO adjust these constants
MIN_NUM_EXAMPLES_IN_QUEUE = 10
NUM_CONSUMING_THREADS = 1


def inputs():
    filenames = build_filename_list()
    filepaths, labels = _load_labelmap()
    filepaths = tf.constant(filepaths)
    labels = tf.constant(labels, dtype=tf.int32)

    # filename_queue = tf.train.string_input_producer(filenames, num_epochs=FLAGS.training_epochs, shuffle=True)
    filename_queue = tf.RandomShuffleQueue(len(filenames), 0, [tf.string, tf.int32], name='filename_queue')
    enqueue_op = filename_queue.enqueue_many([filepaths, labels])
    qr = tf.train.QueueRunner(filename_queue, [enqueue_op])
    tf.train.add_queue_runner(qr)

    example_list = [_read_and_preprocess_image(filename_queue) for _ in xrange(NUM_CONSUMING_THREADS)]

    image_batch, filename_batch = tf.train.shuffle_batch_join(
            example_list,
            batch_size=FLAGS.batch_size,
            capacity=MIN_NUM_EXAMPLES_IN_QUEUE + (NUM_CONSUMING_THREADS + 2) * FLAGS.batch_size,
            min_after_dequeue=MIN_NUM_EXAMPLES_IN_QUEUE,
            shapes=[[224, 224, 3], []]
    )

    return image_batch, util.encode_one_hot(filename_batch, FLAGS.num_classes)


def _load_labelmap():
    if not gfile.Exists(FLAGS.labelmap_path):
        raise ValueError('Label map file not found.')

    filepaths = []
    labels = []

    with open(FLAGS.labelmap_path) as f:
        for line in f:
            fp, lbl = line.split(',')
            filepaths.append(fp)
            labels.append(int(lbl[:-1]))

    return filepaths, labels


def _read_and_preprocess_image(filename_queue):
    filename, label = filename_queue.dequeue()

    # we want to reuse the filename-label-pair for later training steps
    filename_queue.enqueue([filename, label])

    image = tf.read_file(filename)

    # prepare image
    image = tf.image.decode_jpeg(image, channels=3)
    feature_img = tf.cast(image, tf.float32)
    processed_img = preprocess(feature_img)

    return [processed_img, label]
