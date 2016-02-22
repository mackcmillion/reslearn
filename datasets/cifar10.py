import os

import tensorflow as tf

import util
from datasets.dataset import Dataset
from config import FLAGS
from preprocess import preprocess_for_training, preprocess_for_evaluation
from scripts.meanstddev import compute_overall_mean_stddev


class Cifar10(Dataset):
    _NUM_TRAINING_IMAGES = 50000
    _NUM_VALIDATION_IMAGES = 10000

    def __init__(self):
        super(Cifar10, self).__init__(10)
        self._color_data = None

    def pre_graph(self):
        compute_overall_mean_stddev(overwrite=False, num_threads=FLAGS.num_consuming_threads, num_logs=10)
        self._color_data = util.load_meanstddev(FLAGS.mean_stddev_path)

    def preliminary(self):
        pass

    def training_inputs(self):
        filenames = [os.path.join(FLAGS.cifar10_image_path, 'data_batch_%i.bin' % i) for i in xrange(1, 6)]
        filename_queue = tf.train.string_input_producer(filenames, name='training_filename_queue')

        image, label = self._read_image(filename_queue)

        # TODO implement correct image preprocessing for CIFAR-10
        image = preprocess_for_training(image, *self._color_data)

        min_num_examples_in_queue = int(FLAGS.min_frac_examples_in_queue * self._NUM_TRAINING_IMAGES)
        image_batch, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_consuming_threads,
                capacity=min_num_examples_in_queue + (FLAGS.num_consuming_threads + 2) * FLAGS.batch_size,
                min_after_dequeue=min_num_examples_in_queue,
                shapes=[[224, 224, 3], []],
                name='training_example_queue'
        )
        return image_batch, util.encode_one_hot(label_batch, self.num_classes)

    def evaluation_inputs(self, validation_data=True):
        if validation_data:
            filenames = [os.path.join(FLAGS.cifar10_image_path, 'data_batch_%i.bin' % i) for i in xrange(1, 6)]
            num_images = self._NUM_VALIDATION_IMAGES
        else:
            filenames = [os.path.join(FLAGS.cifar10_image_path, 'test_batch.bin')]
            num_images = self._NUM_TRAINING_IMAGES

        filename_queue = tf.train.string_input_producer(filenames, name='validation_filename_queue')

        image, label = self._read_image(filename_queue)

        # TODO implement correct image preprocessing for CIFAR-10
        image = preprocess_for_evaluation(image, *self._color_data[:2])

        min_num_examples_in_queue = int(FLAGS.min_frac_examples_in_queue * num_images)

        example_queue = tf.RandomShuffleQueue(num_images, min_num_examples_in_queue,
                                              dtypes=[tf.float32, tf.int32], shapes=[image.get_shape(), []],
                                              name='validation_filename_queue')
        enqueue_op = example_queue.enqueue([image, label])
        qr = tf.train.QueueRunner(example_queue, [enqueue_op])
        tf.train.add_queue_runner(qr)

        image_10crop, label = example_queue.dequeue()
        return image_10crop, label

    @staticmethod
    def _read_image(filename_queue):
        # copied from
        # https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10_input.py

        # CIFAR-10 specification
        label_bytes = 1
        height = 32
        width = 32
        depth = 3

        image_bytes = height * width * depth
        record_bytes = label_bytes + image_bytes
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
        depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                 [depth, height, width])
        uint8image = tf.transpose(depth_major, [1, 2, 0])
        image = tf.cast(uint8image, tf.float32)

        return image, tf.squeeze(label)
