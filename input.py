import tensorflow as tf
import os

from hyperparams import FLAGS
from image_preprocessing import preprocess

# TODO adjust these constants
MIN_NUM_EXAMPLES_IN_QUEUE = 10
NUM_PRODUCING_THREADS = 1
NUM_CONSUMING_THREADS = 1


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
    _, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    feature_img = tf.cast(image, tf.float32)
    processed_img = preprocess(feature_img)

    # TODO implement label loading
    lbls = tf.zeros([FLAGS.num_classes], dtype=tf.int32)

    return processed_img, lbls
