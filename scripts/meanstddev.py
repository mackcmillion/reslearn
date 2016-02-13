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

    # stddev computation
    variance = tf.Variable([0.0, 0.0, 0.0])
    total = tf.Variable(0.0)

    image = _image_op(filenames)

    stddev_ops = _stddev_ops(image, variance, total, mean, num_threads)
    stddev = _init_and_run_in_loop(stddev_ops, 'standard deviation', _stddev_final_op, (variance, total),
                                   num_logs, len(filenames))

    print 'Computed mean as %s and standard deviation as %s.' % (str(mean), str(stddev))

    print 'Saving to file %s...' % FLAGS.mean_stddev_path
    f = open(FLAGS.mean_stddev_path, 'w')
    mean_str = '[%f,%f,%f]' % (mean[0], mean[1], mean[2])
    stddev_str = '[%f,%f,%f]' % (stddev[0], stddev[1], stddev[2])
    f.write('%s\n%s\n' % (mean_str, stddev_str))
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

    queue = _make_queue([sum_, num], num_threads, 'mean_queue')

    img_sum, img_num = queue.dequeue()
    mean_update = mean.assign_add(img_sum)
    total_update = total.assign_add(img_num)

    return [mean_update, total_update]


def _stddev_ops(image, variance, total, mean, num_threads):
    num = tf.mul(tf.shape(image)[0], tf.shape(image)[1])

    mean_tiled = tf.tile(mean, tf.expand_dims(num, 0))
    mean_tiled = tf.reshape(mean_tiled, tf.pack([tf.shape(image)[0], tf.shape(image)[1], 3]))

    num = tf.cast(num, tf.float32)

    remainders = tf.sub(image, mean_tiled)
    squares = tf.square(remainders)
    sum_of_squares = tf.reduce_sum(squares, [0, 1])

    queue = _make_queue([sum_of_squares, num], num_threads, 'stddev_queue')

    img_sum_sq, img_num = queue.dequeue()
    variance_update = variance.assign_add(img_sum_sq)
    total_update = total.assign_add(img_num)

    return [variance_update, total_update]


def _make_queue(enqueue_ops, num_threads, queue_name):
    queue = tf.FIFOQueue(1000, dtypes=[tf.float32, tf.float32], shapes=[[3], []], name=queue_name)
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


def _stddev_final_op(sum_squares, total):
    total = total - tf.constant(1.0)
    total_3channel = tf.pack([total, total, total])
    variance = tf.div(sum_squares, total_3channel)
    return tf.sqrt(variance)


if __name__ == '__main__':
    compute_overall_mean_stddev(overwrite=True, num_threads=4)
