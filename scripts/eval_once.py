import math

import tensorflow as tf
from datetime import datetime as dt

from config import FLAGS
from datasets.yelp import Yelp
from models.resnet18 import ResNet18

tf.app.flags.DEFINE_string('checkpoint', '/home/max/checkpoints/yelp_resnet_18_2016-04-10_15-04-49/resnet-18.ckpt-901000',
                           """The checkpoint to compute test error from.""")


tf.app.flags.DEFINE_string('target_filepath', '/home/max/prediction_map',
                           """File to write the result to.""")


def evaluate(dataset, model):

    dataset.pre_graph()

    with tf.Graph().as_default():
        # input and evaluation procedure
        images, true_labels, filenames = dataset.evaluation_inputs()
        predictions = model.inference(images, dataset.num_classes, False)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            _eval_once(sess, coord, saver, tf.app.flags.FLAGS.checkpoint, dataset, predictions, filenames, true_labels)

            coord.request_stop()
            coord.join(threads)


def _eval_once(sess, coord, saver, read_checkpoint_path, dataset, pred_op, filenames, true_labels):
    saver.restore(sess, read_checkpoint_path)
    print '%s - Restored model from checkpoint %s.' % (dt.now(), read_checkpoint_path)

    results = {}

    try:
        num_iter = int(math.ceil((1.0 * dataset.num_evaluation_images) / FLAGS.batch_size))
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run(pred_op)
            for filename, true_label, prediction in zip(filenames, true_labels, predictions):
                results[filename] = (true_label, prediction)
            step += 1
            print '%s - %d of %d images' % \
                  (dt.now(), step * FLAGS.batch_size, dataset.num_evaluation_images)

    except Exception as e:  # pylint: disable=broad-except
        print e
        coord.request_stop(e)

    with open(tf.app.flags.FLAGS.target_filepath, 'w') as target_file:
        target_file.write('image,true_labels,predictions')
        for result in results:
            true_label, prediction = results[result]
            target_file.write('%s,%s,%s' % (result, str(true_label), str(prediction)))


def main(argv=None):  # pylint: disable=unused-argument
    evaluate(Yelp(), ResNet18())


if __name__ == '__main__':
    tf.app.run()
