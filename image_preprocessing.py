import tensorflow as tf
import random


def rescale_and_crop(image):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_shorter_edge = tf.constant(random.choice(xrange(256, 480 + 1)))

    height_smaller_than_width = tf.less_equal(height, width)
    new_height, new_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, (width * new_shorter_edge) / height),
        lambda: (((height * new_shorter_edge) / width), new_shorter_edge)
    )

    image = tf.image.resize_images(image, new_height, new_width)
    image = tf.image.random_flip_left_right(image)
    return tf.image.random_crop(image, [224, 224])


def augment_colors(image, mean):
    image = image - mean

    # eigens = tf.self_adjoint_eig(image)
    # eigenvalues = eigens[0]
    # eigenvectors = eigens[1:]
    #
    # print tf.shape(eigenvalues)
    # print tf.shape(eigenvectors)

    return image


def preprocess(image):
    size_adjusted = rescale_and_crop(image)

    # FIXME implement mean calculation
    mean = tf.zeros(size_adjusted.get_shape())

    return augment_colors(size_adjusted, mean)
