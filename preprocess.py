import tensorflow as tf
from tensorflow.python.platform import gfile

from hyperparams import FLAGS


def augment_scale(image):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_shorter_edge = tf.random_uniform([], minval=256, maxval=480 + 1, dtype=tf.int32)

    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
        lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
    )

    # workaround since tf.image.resize_images() does not work
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, tf.pack(new_height_and_width), name='RESIZE')
    image = tf.squeeze(image, [0])

    return tf.image.random_crop(image, [224, 224])


def _compute_longer_edge(shorter, longer, new_shorter):
    return (longer * new_shorter) / shorter


def augment_colors(image):
    image = _normalize_colors(image)

    # eigens = tf.self_adjoint_eig(image)
    # eigenvalues = eigens[0]
    # eigenvectors = eigens[1:]
    #
    # print tf.shape(eigenvalues)
    # print tf.shape(eigenvectors)

    return image


def _normalize_colors(image):
    # load precomputed mean/stddev
    if not gfile.Exists(FLAGS.mean_stddev_path):
        print 'Mean/stddev file not found. Computing. This might potentially take a long time...'

    assert gfile.Exists(FLAGS.mean_stddev_path)
    mean_stddev_string = open(FLAGS.mean_stddev_path, 'r').read().split('\n')
    mean_str = mean_stddev_string[0][1:-1].split(' ')
    stddev_str = mean_stddev_string[1][1:-1].split(' ')

    mean = tf.constant([float(mean_str[0]), float(mean_str[1]), float(mean_str[2])], dtype=tf.float32)
    stddev = tf.constant([float(stddev_str[0]), float(stddev_str[1]), float(stddev_str[2])], dtype=tf.float32)

    total_pixels = tf.shape(image)[0] * tf.shape(image)[1]
    image_shape = tf.pack([tf.shape(image)[0], tf.shape(image)[1], 3])

    mean = tf.reshape(tf.tile(mean, tf.expand_dims(total_pixels, 0)), image_shape)
    stddev = tf.reshape(tf.tile(stddev, tf.expand_dims(total_pixels, 0)), image_shape)

    # final normalization
    return (image - mean) / stddev


def preprocess(image):
    size_adjusted = augment_scale(image)

    return augment_colors(size_adjusted)
