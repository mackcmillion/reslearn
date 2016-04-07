import tensorflow as tf
from tensorflow.python.platform import gfile


def unoptimized_weight_variable(shape, name, wd, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name + '_weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def encode_one_hot(label_batch, num_labels):
    sparse_labels = tf.reshape(label_batch, [-1, 1])
    derived_size = tf.shape(label_batch)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, num_labels])
    return tf.sparse_to_dense(concated, outshape, sparse_values=1.0, default_value=0.0)


def encode_k_hot_python(labels, num_labels):
    k_hot_array = []
    for instance_labels in labels:
        instance_k_hot = [0.0 for _ in xrange(num_labels)]
        for label in instance_labels:
            instance_k_hot[label] = 1.0
        k_hot_array.append(instance_k_hot)
    return k_hot_array


def format_time_hhmmss(timediff):
    hours, remainder = divmod(timediff, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%dh %02dm %02ds' % (hours, minutes, seconds)


def load_meanstddev(path):
    # load precomputed mean/stddev
    if not gfile.Exists(path):
        raise ValueError('Mean/stddev file not found.')

    assert gfile.Exists(path)
    mean_stddev_string = open(path, 'r').read().split('\n')
    mean_str = mean_stddev_string[0][1:-1].split(',')
    stddev_str = mean_stddev_string[1][1:-1].split(',')
    eigval_str = mean_stddev_string[2][1:-1].split(',')
    eigvecs_str = mean_stddev_string[3][1:-1].split(' ')

    mean = [float(mean_str[0]), float(mean_str[1]), float(mean_str[2])]
    stddev = [float(stddev_str[0]), float(stddev_str[1]), float(stddev_str[2])]
    eigvals = [float(eigval_str[0]), float(eigval_str[1]), float(eigval_str[2])]
    eigvecs = []
    for eigvec_str in eigvecs_str:
        eigvec = eigvec_str[1:-1].split(',')
        eigvecs.append([float(eigvec[0]), float(eigvec[1]), float(eigvec[2])])
    return mean, stddev, eigvals, eigvecs


def replicate_to_image_shape(image, t, channels=1):
    img_shape = tf.shape(image)
    multiples = tf.pack([img_shape[0], img_shape[1], channels])
    t = tf.expand_dims(tf.expand_dims(t, 0), 0)
    t = tf.tile(t, multiples)
    return t


# transforms color values to values relative to the channel maximum (256)
def absolute_to_relative_colors(image):
    maximum = replicate_to_image_shape(image, tf.constant([256], dtype=tf.float32, shape=[1]), channels=3)
    return tf.div(image, maximum)


def extract_global_step(path):
    return int(path.split('/')[-1].split('-')[-1])


DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'


def _get_lower_triangular_mask(batch_size, num_classes):
    indices = tf.range(0, num_classes)
    indices = tf.expand_dims(tf.expand_dims(indices, 0), 0)
    indices_matrix = tf.tile(indices, [batch_size, num_classes, 1])
    indices_matrix_transposed = tf.transpose(indices_matrix, [0, 2, 1])

    return tf.less_equal(indices_matrix, indices_matrix_transposed)


def mask_lower_triangular(boolean_matrix):
    shape = boolean_matrix.get_shape().as_list()
    lower_triangular = _get_lower_triangular_mask(shape[0], shape[1])

    return tf.logical_and(boolean_matrix, lower_triangular)


def fan_out_to_matrix(vectorbatch):
    matrix = tf.expand_dims(vectorbatch, 1)
    matrix = tf.tile(matrix, [1, vectorbatch.get_shape()[1].value, 1])
    return matrix, tf.transpose(matrix, [0, 2, 1])


def mll_error(predictions, true_labels):
    num_classes = true_labels.get_shape()[1].value
    true_labels_bool = tf.cast(true_labels, tf.bool)

    num_positives = tf.reduce_sum(true_labels, [1])
    num_negatives = num_classes - num_positives
    normalizing_factor = 1.0 / (num_positives * num_negatives)

    # trick to compute a mask that masks out all elements not satisfying
    # (k, l) in Y_i x notY_i
    tl_matrix, tl_matrix_transposed = fan_out_to_matrix(true_labels_bool)
    setproduct = tf.logical_xor(tl_matrix, tl_matrix_transposed)
    y_x_noty_mask = tf.logical_and(setproduct, tl_matrix)

    pred_matrix, pred_matrix_transposed = fan_out_to_matrix(predictions)

    def op_on_pred_pair(matrix, matrix_transposed):
        return tf.exp(tf.neg(matrix - matrix_transposed))

    valid_vals = tf.mul(op_on_pred_pair(pred_matrix, pred_matrix_transposed), tf.cast(y_x_noty_mask, tf.float32))
    summed_error = tf.reduce_sum(valid_vals, [1, 2])
    normalized_error = normalizing_factor * summed_error

    return tf.reduce_sum(normalized_error)
