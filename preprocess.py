import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.ops import control_flow_ops as cf

from hyperparams import FLAGS

MEAN = None
STDDEV = None
EIGVALS = None
EIGVECS = None


def resize_random(image, minval, maxval_inc):
    new_shorter_edge_tensor = tf.random_uniform([], minval=minval, maxval=maxval_inc + 1, dtype=tf.int32)
    return _resize_aux(image, new_shorter_edge_tensor)


def resize(image, new_shorter_edge):
    new_shorter_edge_tensor = tf.constant(new_shorter_edge, dtype=tf.int32)
    return _resize_aux(image, new_shorter_edge_tensor)


def _resize_aux(image, new_shorter_edge_tensor):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = cf.cond(
            height_smaller_than_width,
            lambda: (new_shorter_edge_tensor, _compute_longer_edge(height, width, new_shorter_edge_tensor)),
            lambda: (_compute_longer_edge(width, height, new_shorter_edge_tensor), new_shorter_edge_tensor)
    )

    # workaround since tf.image.resize_images() does not work
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, tf.pack(new_height_and_width))
    return tf.squeeze(image, [0])


def random_flip(image):
    return tf.image.random_flip_left_right(image)


def random_crop_to_square(image, size):
    return tf.image.random_crop(image, [size, size])


def _compute_longer_edge(shorter, longer, new_shorter):
    return (longer * new_shorter) / shorter


def color_noise(image):
    alpha = tf.random_normal([3], 0.0, 0.1, dtype=tf.float32)
    q = tf.matmul(EIGVECS, tf.expand_dims(alpha * EIGVALS, 1))
    q = tf.squeeze(q)

    return image + _replicate_to_image_shape(image, q)


def normalize_colors(image):
    mean = _replicate_to_image_shape(image, MEAN)
    stddev = _replicate_to_image_shape(image, STDDEV)

    return (image - mean) / stddev


def _replicate_to_image_shape(image, t):
    img_shape = tf.shape(image)
    multiples = tf.pack([img_shape[0], img_shape[1], 1])
    t = tf.expand_dims(tf.expand_dims(t, 0), 0)
    t = tf.tile(t, multiples)
    return t


def preprocess_for_training(image):
    if not MEAN:
        _load_meanstddev()
    image = resize_random(image, 256, 480)
    # swapped cropping and flipping because flip needs image shape to be fully defined - should not make a difference
    image = random_crop_to_square(image, 224)
    image = random_flip(image)
    image = color_noise(image)
    image = normalize_colors(image)
    return image


def ten_crop(image):
    image = resize(image, 256)
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    flipped_image = tf.image.flip_left_right(image)

    crops = _extract_5crop(image)
    flipped_crops = _extract_5crop(flipped_image)
    return tf.concat(0, [crops, flipped_crops])


def _extract_5crop(image):
    return tf.image.extract_glimpse(image, [224, 224],
                                    [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
                                    centered=True, normalized=True)


def preprocess_for_validation(image):
    # TODO should one normalize here?
    return ten_crop(image)


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
