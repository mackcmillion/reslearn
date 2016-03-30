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
