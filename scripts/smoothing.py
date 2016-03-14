import os

import tensorflow as tf
import csv
from collections import OrderedDict

from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_string(
        'csv_path',
        '/home/max/Studium/Kurse/BA2/results/csv/run_resnet_20_2016-03-10_22-45-04_test,tag_test_error_raw.csv',
        """Path to the CSV file to apply moving average to.""")

tf.app.flags.DEFINE_string(
    'summary_path',
    '/home/max/Studium/Kurse/BA2/results/summaries',
    """Path to write the resulting summary."""
)


def apply_moving_average_to_csv():

    dirname = 'smoothing' + ('_%s' % tf.app.flags.FLAGS.csv_path.split('/')[-1].split('.')[0])
    summary_path = os.path.join(tf.app.flags.FLAGS.summary_path, dirname)

    if not gfile.Exists(summary_path):
        gfile.MkDir(summary_path)

    d = _read_csv()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    p = tf.placeholder(tf.float32, shape=[])
    test_err = tf.Variable(1.0, trainable=False)
    test_err_assign = test_err.assign(p)
    avg_op = averaging_op(test_err, test_err_assign)

    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

    sess.run(init_op)

    summary_writer = tf.train.SummaryWriter(summary_path, sess.graph_def)

    for step in d:
        sess.run(avg_op, feed_dict={p: d[step]})
        summary = sess.run(summary_op)
        summary_writer.add_summary(summary, step)

    sess.close()


def _read_csv():
    d = OrderedDict()
    with open(tf.app.flags.FLAGS.csv_path, 'r') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            d[int(row['Step'])] = float(row['Value'])
    return d


def _add_moving_average_summary(test_err):
    train_err_avg_obj = tf.train.ExponentialMovingAverage(0.9, name='test_err_avg')
    train_err_avg_op = train_err_avg_obj.apply([test_err])

    averaged = train_err_avg_obj.average(test_err)
    tf.scalar_summary('test_error_averaged', averaged)

    return train_err_avg_op


def averaging_op(train_err, test_err_assign):
    test_err_avg_op = _add_moving_average_summary(train_err)
    with tf.control_dependencies([test_err_assign]):
        with tf.control_dependencies([test_err_avg_op]):
            avg_op = tf.no_op()
    return avg_op


def main(argv=None):  # pylint: disable=unused-argument
    apply_moving_average_to_csv()


if __name__ == '__main__':
    tf.app.run()
