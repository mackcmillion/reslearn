import csv
from collections import OrderedDict
from learningrate import dynamic_decay_default

import tensorflow as tf

tf.app.flags.DEFINE_string(
        'csv_path',
        '/home/max/Studium/Kurse/BA2/results/csv/run_resnet_20_adam_100000_2016-03-25_19-13-18,tag_training_error_raw.csv',
        """Path to the CSV file to apply moving average to.""")

# tf.app.flags.DEFINE_string(
#         'summary_path',
#         '/home/max/Studium/Kurse/BA2/summaries',
#         """Path to write the resulting summary."""
# )


def main(argv=None):
    d = _read_csv()
    lr = 0.1

    for step in d:
        lr = dynamic_decay_default(lr, step, d[step])
        print '%i: %.3f' % (step, lr)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    #
    # p = tf.placeholder(tf.float32, shape=[])
    # der = tf.placeholder(tf.float32, shape=[])
    # train_err = tf.Variable(1.0, trainable=False)
    # derivative = tf.Variable(0.0, trainable=False)
    # train_err_assign = train_err.assign(p)
    # avg_op = averaging_op(train_err, train_err_assign, 'training_error')
    # deriv_assign = derivative.assign(der)
    # der_avg_op = averaging_op(derivative, deriv_assign, 'derivative')
    #
    # tf.scalar_summary('training_error', train_err)
    # tf.scalar_summary('derivative', derivative)
    #
    # summary_op = tf.merge_all_summaries()
    # init_op = tf.initialize_all_variables()
    #
    # sess.run(init_op)
    #
    # summary_writer = tf.train.SummaryWriter(tf.app.flags.FLAGS.summary_path, sess.graph_def)
    #
    # previous = 2.34
    # accum = []
    # for step in d:
    #
    #     if step % 2000 == 0:
    #         avg = sum(accum) / len(accum)
    #         mini = min(accum)
    #         sess.run([avg_op, der_avg_op], feed_dict={p: avg, der: mini})
    #
    #         summary = sess.run(summary_op)
    #         summary_writer.add_summary(summary, step)
    #
    #         accum = []
    #
    #     accum.append(d[step])
    #
    # sess.close()


def _add_moving_average_summary(test_err, name):
    train_err_avg_obj = tf.train.ExponentialMovingAverage(0.9999, name='%s_avg' % name)
    train_err_avg_op = train_err_avg_obj.apply([test_err])

    averaged = train_err_avg_obj.average(test_err)
    tf.scalar_summary('%s_averaged' % name, averaged)

    return train_err_avg_op


def averaging_op(train_err, test_err_assign, name):
    test_err_avg_op = _add_moving_average_summary(train_err, name)
    with tf.control_dependencies([test_err_assign]):
        with tf.control_dependencies([test_err_avg_op]):
            avg_op = tf.no_op()
    return avg_op


def _read_csv():
    d = OrderedDict()
    with open(tf.app.flags.FLAGS.csv_path, 'r') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            d[int(row['Step'])] = float(row['Value'])
    return d

if __name__ == '__main__':
    tf.app.run()
