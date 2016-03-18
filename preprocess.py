import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as cf

from util import replicate_to_image_shape


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


def _compute_longer_edge(shorter, longer, new_shorter):
    return (longer * new_shorter) / shorter


def random_flip(image):
    return tf.image.random_flip_left_right(image)


def random_crop_to_square(image, size):
    return tf.random_crop(image, [size, size, 3])


def evenly_pad_zeros(image, num):
    img_shape = image.get_shape().as_list()
    height = img_shape[0]
    width = img_shape[1]
    new_height = height + 2 * num
    new_width = width + 2 * num
    return tf.image.resize_image_with_crop_or_pad(image, new_height, new_width)


def color_noise(image, eigvals, eigvecs):
    eigvals = tf.constant(eigvals, dtype=tf.float32)
    eigvecs = tf.constant(eigvecs, dtype=tf.float32, shape=[3, 3])

    alpha = tf.random_normal([3], 0.0, 0.1, dtype=tf.float32)
    q = tf.matmul(eigvecs, tf.expand_dims(alpha * eigvals, 1))
    q = tf.squeeze(q)

    return image + replicate_to_image_shape(image, q)


def normalize_colors(image, mean, stddev):
    mean = tf.constant(mean, dtype=tf.float32)
    stddev = tf.constant(stddev, dtype=tf.float32)

    mean = replicate_to_image_shape(image, mean)
    stddev = replicate_to_image_shape(image, stddev)

    return (image - mean) / stddev


def single_crop(image, size):
    # workaround for getting the center square crop of an image with not fully defined shape
    image = tf.expand_dims(image, 0)
    return tf.image.extract_glimpse(image, [size, size], [[0.0, 0.0]], centered=True, normalized=True)


def ten_crop(image):
    image = resize(image, 256)
    image = single_crop(image, 256)
    flipped_image = tf.image.flip_left_right(tf.squeeze(image))
    flipped_image = tf.expand_dims(flipped_image, 0)

    crops = _extract_5crop(image)
    flipped_crops = _extract_5crop(flipped_image)
    return tf.concat(0, [crops, flipped_crops])


def _extract_5crop(image):
    if image.get_shape().ndims < 4:
        image = tf.expand_dims(image, 0)
    tiled = tf.tile(image, [5, 1, 1, 1])
    return tf.image.extract_glimpse(tiled, [224, 224],
                                    [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
                                    centered=True, normalized=True)
