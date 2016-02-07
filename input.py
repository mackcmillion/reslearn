import tensorflow as tf
import os

from hyperparams import FLAGS
from image_preprocessing import preprocess

# TODO adjust these constants
MIN_NUM_EXAMPLES_IN_QUEUE = 10
NUM_THREADS = 4


def inputs():
    filenames = [os.path.join(FLAGS.train_dir, filename) for filename in os.listdir(FLAGS.train_dir)
                 if os.path.isfile(os.path.join(FLAGS.train_dir, filename))]
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)

    image, lbls = _read_image(filename_queue)
    feature_img = tf.cast(image, tf.float32)
    processed_img = preprocess(feature_img)

    return _generate_batch(processed_img, lbls)


def _read_image(filename_queue):
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tensor = tf.image.decode_jpeg(value, channels=3)

    # TODO implement label loading
    lbls = tf.constant(1, dtype=tf.int32)

    return image_tensor, lbls


def _generate_batch(image, lbls):
    image_batch, label_batch = tf.train.shuffle_batch(
            [image, lbls],
            batch_size=FLAGS.batch_size,
            num_threads=NUM_THREADS,
            capacity=MIN_NUM_EXAMPLES_IN_QUEUE + 3 * FLAGS.batch_size,
            min_after_dequeue=MIN_NUM_EXAMPLES_IN_QUEUE
    )

    print "image_batch shape: " + str(image_batch.get_shape())
    print "label_batch shape: " + str(label_batch.get_shape())

    return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])


if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=sess)
    print sess.run(inputs())
