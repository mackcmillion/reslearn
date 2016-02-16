import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.ops import control_flow_ops as cf

from hyperparams import FLAGS

MEAN = None
STDDEV = None
EIGVALS = None
EIGVECS = None


def augment_scale(image):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_shorter_edge = tf.random_uniform([], minval=256, maxval=480 + 1, dtype=tf.int32)

    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = cf.cond(
            height_smaller_than_width,
            lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
            lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
    )

    # workaround since tf.image.resize_images() does not work
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, tf.pack(new_height_and_width))
    image = tf.squeeze(image, [0])

    image = tf.image.random_flip_left_right(image)
    return tf.image.random_crop(image, [224, 224])


def _compute_longer_edge(shorter, longer, new_shorter):
    return (longer * new_shorter) / shorter


def augment_colors(image):
    image = _color_noise(image)
    image = _normalize_colors(image)

    return image


def _color_noise(image):

    alpha = tf.random_normal([3], 0.0, 0.1, dtype=tf.float32)
    q = tf.matmul(EIGVECS, tf.expand_dims(alpha * EIGVALS, 1))
    q = tf.squeeze(q)

    return image + _replicate_to_image_shape(image, q)


def _normalize_colors(image):
    mean = _replicate_to_image_shape(image, MEAN)
    stddev = _replicate_to_image_shape(image, STDDEV)

    # final normalization
    return (image - mean) / stddev


def _replicate_to_image_shape(image, t):
    img_shape = tf.shape(image)
    multiples = tf.pack([img_shape[0], img_shape[1], 1])
    t = tf.expand_dims(tf.expand_dims(t, 0), 0)
    t = tf.tile(t, multiples)
    return t


def preprocess(image):
    if not MEAN:
        _load_meanstddev()
    size_adjusted = augment_scale(image)
    return augment_colors(size_adjusted)


def _load_meanstddev():
    global MEAN, STDDEV, EIGVALS, EIGVECS
    # load precomputed mean/stddev
    if not gfile.Exists(FLAGS.mean_stddev_path):
        raise ValueError('Mean/stddev file not found.')

    assert gfile.Exists(FLAGS.mean_stddev_path)
    mean_stddev_string = open(FLAGS.mean_stddev_path, 'r').read().split('\n')
    mean_str = mean_stddev_string[0][1:-1].split(',')
    stddev_str = mean_stddev_string[1][1:-1].split(',')
    eigval_str = mean_stddev_string[2][1:-1].split(',')
    eigvecs_str = mean_stddev_string[3][1:-1].split(' ')

    MEAN = tf.constant([float(mean_str[0]), float(mean_str[1]), float(mean_str[2])], dtype=tf.float32)
    STDDEV = tf.constant([float(stddev_str[0]), float(stddev_str[1]), float(stddev_str[2])], dtype=tf.float32)
    EIGVALS = tf.constant([float(eigval_str[0]), float(eigval_str[1]), float(eigval_str[2])], dtype=tf.float32)
    eigvecs = []
    for eigvec_str in eigvecs_str:
        eigvec = eigvec_str[1:-1].split(',')
        eigvecs.append([float(eigvec[0]), float(eigvec[1]), float(eigvec[2])])
    EIGVECS = tf.constant(eigvecs, dtype=tf.float32, shape=[3, 3])
