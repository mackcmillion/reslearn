import os

import tensorflow as tf
from tensorflow.python.platform import gfile

from hyperparams import FLAGS
from preprocess import preprocess_for_validation


def validation_inputs():
    fps, labels = _load_validation_labelmap()
    filepaths = tf.constant(fps)
    labels = tf.constant(labels, dtype=tf.int32)

    filename_queue = tf.FIFOQueue(len(fps), [tf.string, tf.int32], name='validation_filename_queue')
    enqueue_op = filename_queue.enqueue_many([filepaths, labels])
    qr = tf.train.QueueRunner(filename_queue, [enqueue_op])
    tf.train.add_queue_runner(qr)

    example_queue = tf.FIFOQueue(len(filepaths), [tf.float32, tf.int32], name='validation_example_queue')
    enqueue_op_ex = example_queue.enqueue(_read_and_preprocess_image_for_validation(filename_queue))
    qr_ex = tf.train.QueueRunner(example_queue, [enqueue_op_ex] * FLAGS.num_consuming_threads)
    tf.train.add_queue_runner(qr_ex)

    image_10crop, label = example_queue.dequeue()
    # do not one-hot-encode label here
    return image_10crop, label


def _load_validation_labelmap():
    if not gfile.Exists(FLAGS.validation_set):
        raise ValueError('Validation label map not found.')

    filepaths = []
    labels = []

    with open(FLAGS.validation_set) as f:
        for line_no, line in enumerate(f):
            fp = os.path.join(FLAGS.validation_images, 'ILSVRC2012_val_%08d.JPEG' % line_no)
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
