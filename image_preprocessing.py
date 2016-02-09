import tensorflow as tf


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
    size_adjusted = augment_scale(image)

    # FIXME implement mean calculation
    mean = tf.zeros(size_adjusted.get_shape())

    return augment_colors(size_adjusted, mean)
