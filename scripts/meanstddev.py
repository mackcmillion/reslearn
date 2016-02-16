import tensorflow as tf
from tensorflow.python.platform import gfile

from hyperparams import FLAGS
from scripts.labelmap import build_filename_list


def compute_overall_mean_stddev(overwrite=False, num_threads=1, num_logs=10):
    if gfile.Exists(FLAGS.mean_stddev_path):
        print 'Mean/stddev file already exists.'
        if overwrite:
            print 'Overwriting file...'
            gfile.Remove(FLAGS.mean_stddev_path)
        else:
            print 'Nothing to do here.'
            return
        print

    print 'Building filename list...'
    filenames = build_filename_list()

    mean = tf.Variable([0.0, 0.0, 0.0])
    total = tf.Variable(0.0)

    image = _image_op(filenames)

    # mean computation
    mean_ops = _mean_ops(image, mean, total, num_threads)
    mean = _init_and_run_in_loop(mean_ops, 'mean', _mean_final_op, (mean, total), num_logs, len(filenames))

    tf.reset_default_graph()

    # stddev and PCA computation
    covariance = tf.Variable(tf.zeros([3, 3], dtype=tf.float32))
    total = tf.Variable(0.0)

    image = _image_op(filenames)

    covariance_ops = _covariance_ops(image, covariance, total, mean, num_threads)
    stddev, eigvals, eigvecs = _init_and_run_in_loop(covariance_ops, 'standard deviation', _covariance_final_ops,
                                                     (covariance, total), num_logs, len(filenames))

    print 'Computed mean as %s and standard deviation as %s.' % (str(mean), str(stddev))

    print 'Saving to file %s...' % FLAGS.mean_stddev_path
    f = open(FLAGS.mean_stddev_path, 'w')
    mean_str = '[%f,%f,%f]' % (mean[0], mean[1], mean[2])
    stddev_str = '[%f,%f,%f]' % (stddev[0], stddev[1], stddev[2])
    eigval_str = '[%f,%f,%f]' % (eigvals[0], eigvals[1], eigvals[2])
    eigvec_str = '[' + ' '.join('[%f,%f,%f]' % (eigvec[0], eigvec[1], eigvec[2]) for eigvec in eigvecs) + ']'
    f.write('%s\n%s\n%s\n%s' % (mean_str, stddev_str, eigval_str, eigvec_str))
    f.close()

    print 'Done.'


def _image_op(filenames):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    return tf.cast(image, tf.float32)


def _mean_ops(image, mean, total, num_threads):
    sum_ = tf.reduce_sum(image, [0, 1])
    num = tf.shape(image)[0] * tf.shape(image)[1]
    num = tf.cast(num, tf.float32)

    queue = _make_queue([sum_, num], [[3], []], num_threads, 'mean_queue')

    img_sum, img_num = queue.dequeue()
    mean_update = mean.assign_add(img_sum)
    total_update = total.assign_add(img_num)

    return [mean_update, total_update]


def _covariance_ops(image, covariance, total, mean, num_threads):
    num = tf.mul(tf.shape(image)[0], tf.shape(image)[1])

    mean_tiled = tf.tile(mean, tf.expand_dims(num, 0))
    mean_tiled = tf.reshape(mean_tiled, tf.pack([tf.shape(image)[0], tf.shape(image)[1], 3]))

    num = tf.cast(num, tf.float32)

    remainders = tf.sub(image, mean_tiled)
    remainders_stack = tf.pack([remainders, remainders, remainders])
    remainders_stack_transposed = tf.transpose(remainders_stack, [3, 1, 2, 0])
    pseudo_squares = tf.mul(remainders_stack, remainders_stack_transposed)
    sum_of_squares = tf.reduce_sum(pseudo_squares, [1, 2])

    queue = _make_queue([sum_of_squares, num], [[3, 3], []], num_threads, 'covariance_queue')

    img_sum_sq, img_num = queue.dequeue()
    covariance_update = covariance.assign_add(img_sum_sq)
    total_update = total.assign_add(img_num)

    return [covariance_update, total_update]


def _make_queue(enqueue_ops, shapes, num_threads, queue_name):
    queue = tf.FIFOQueue(1000, dtypes=[tf.float32, tf.float32], shapes=shapes, name=queue_name)
    enqueue_op = queue.enqueue(enqueue_ops)
    qr = tf.train.QueueRunner(queue, [enqueue_op] * num_threads)
    tf.train.add_queue_runner(qr)
    return queue


def _init_and_run_in_loop(op, computing_print, final_op, final_op_args, num_logs, num_files):
    log_mod = num_files / num_logs

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_no = 0
    print 'Computing %s...' % computing_print
    try:
        while not coord.should_stop():
            if image_no != 0 and image_no % log_mod == 0:
                print '%i/%i of images processed.' % (int(image_no / log_mod), num_logs)
            sess.run(op)
            image_no += 1

    except tf.errors.OutOfRangeError:
        print 'All images processed for %s computation.' % computing_print
    finally:
        coord.request_stop()

    coord.join(threads)

    result = sess.run(final_op(*final_op_args))
    sess.close()
    return result


def _mean_final_op(sum_, total):
    total_3channel = tf.pack([total, total, total])
    return tf.div(sum_, total_3channel)


def _covariance_final_ops(sum_squares, total):
    # http://www.johnloomis.org/ece563/notes/covar/covar.html
    total = total - tf.constant(1.0)
    total_3x3 = tf.reshape(tf.tile(tf.expand_dims(total, 0), [9]), [3, 3])
    covariance = tf.div(sum_squares, total_3x3)

    variance = tf.gather(tf.reshape(covariance, [-1]), [0, 3, 6])

    # eigenvalues and eigenvectors for PCA
    eigens = tf.self_adjoint_eig(covariance)
    eigenvalues = tf.slice(eigens, [0, 0], [-1, 1])
    eigenvectors = tf.slice(eigens, [1, 0], [-1, -1])

    return tf.sqrt(variance), eigenvalues, eigenvectors


if __name__ == '__main__':
    compute_overall_mean_stddev(overwrite=True, num_threads=4)