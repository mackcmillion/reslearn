import os

import tensorflow as tf

import util
from datasets.dataset import Dataset
from config import FLAGS
from preprocess import evenly_pad_zeros, random_flip, random_crop_to_square, normalize_colors
from scripts.meanstddev import compute_overall_mean_stddev


class Cifar10(Dataset):
    _NUM_TRAINING_IMAGES = 50000
    _NUM_VALIDATION_IMAGES = 10000

    def __init__(self):
        super(Cifar10, self).__init__('cifar10', 10)
        self._color_data = None

    def pre_graph(self):
        compute_overall_mean_stddev(overwrite=False, num_threads=FLAGS.num_consuming_threads, num_logs=10)
        self._color_data = util.load_meanstddev(FLAGS.cifar10_mean_stddev_path)

    def preliminary(self):
        pass

    def training_inputs(self):
        return self._inputs(True)

    def evaluation_inputs(self):
        return self._inputs(False)

    def _inputs(self, is_training):
        if is_training:
            filenames = [os.path.join(FLAGS.cifar10_image_path, 'data_batch_%i.bin' % i) for i in xrange(1, 6)]
        else:
            filenames = [os.path.join(FLAGS.cifar10_image_path, 'test_batch.bin')]
        filename_queue = tf.train.string_input_producer(filenames, name='%s_filename_queue' %
                                                                        'training' if is_training else 'evaluation')

        image, label = self._read_image(filename_queue)

        if is_training:
            image = self._preprocess_for_training(image)
        else:
            image = self._preprocess_for_evaluation(image)

        min_num_examples_in_queue = int(FLAGS.min_frac_examples_in_queue *
                                        self._NUM_TRAINING_IMAGES if is_training else self._NUM_VALIDATION_IMAGES)
        image_batch, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_consuming_threads,
                capacity=min_num_examples_in_queue + (FLAGS.num_consuming_threads + 2) * FLAGS.batch_size,
                min_after_dequeue=min_num_examples_in_queue,
                shapes=[[32, 32, 3], []],
                name='%s_example_queue' % 'training' if is_training else 'evaluation'
        )

        if is_training:
            label_batch = util.encode_one_hot(label_batch, self.num_classes)
        return image_batch, label_batch

    def _preprocess_for_training(self, image):
        image = normalize_colors(image, *self._color_data[:2])
        image = random_flip(image)
        image = evenly_pad_zeros(image, 4)
        image = random_crop_to_square(image, 32)
        return image

    def _preprocess_for_evaluation(self, image):
        return normalize_colors(image, *self._color_data[:2])

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
