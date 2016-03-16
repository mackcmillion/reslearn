import tensorflow as tf
from tensorflow.python.platform import gfile

import util
from config import FLAGS
from datasets.dataset import Dataset
from preprocess import resize_random, random_crop_to_square, random_flip, color_noise, \
    normalize_colors, ten_crop
from scripts.labelmap import create_label_map_file
from scripts.meanstddev import compute_overall_mean_stddev


class ImageNet(Dataset):
    def __init__(self):
        # FIXME extend to original 1000 classes of ImageNet
        super(ImageNet, self).__init__('imagenet', 2)
        self._color_data = None

    def pre_graph(self):
        compute_overall_mean_stddev(overwrite=False, num_threads=FLAGS.num_consuming_threads, num_logs=10)
        self._color_data = util.load_meanstddev(FLAGS.mean_stddev_path)
        create_label_map_file(overwrite=False)

    def preliminary(self):
        pass

    def training_inputs(self):
        fps, labels = self._load_training_labelmap()
        filepaths = tf.constant(fps)
        labels = tf.constant(labels, dtype=tf.int32)

        min_num_examples_in_queue = int(FLAGS.min_frac_examples_in_queue * len(fps))

        filename_queue = tf.RandomShuffleQueue(len(fps), min_num_examples_in_queue, [tf.string, tf.int32],
                                               name='training_filename_queue')
        enqueue_op = filename_queue.enqueue_many([filepaths, labels])
        qr = tf.train.QueueRunner(filename_queue, [enqueue_op])
        tf.train.add_queue_runner(qr)

        example_list = [self._read_and_preprocess_image_for_training(filename_queue) for _ in
                        xrange(FLAGS.num_consuming_threads)]

        image_batch, label_batch = tf.train.shuffle_batch_join(
                example_list,
                batch_size=FLAGS.batch_size,
                capacity=min_num_examples_in_queue + (FLAGS.num_consuming_threads + 2) * FLAGS.batch_size,
                min_after_dequeue=min_num_examples_in_queue,
                shapes=[[224, 224, 3], []],
                name='training_example_queue'
        )

        return image_batch, util.encode_one_hot(label_batch, self.num_classes)

    @staticmethod
    def _load_training_labelmap():
        if not gfile.Exists(FLAGS.training_set):
            raise ValueError('Training label map file not found.')

        filepaths = []
        labels = []

        with open(FLAGS.training_set) as f:
            for line in f:
                fp, lbl = line.split(',')
                filepaths.append(fp)
                labels.append(int(lbl[:-1]))

        return filepaths, labels

    def _read_and_preprocess_image_for_training(self, filename_queue):
        filename, label = filename_queue.dequeue()

        # we want to reuse the filename-label-pair for later training steps
        filename_queue.enqueue([filename, label])

        image = tf.read_file(filename)

        # prepare image
        image = tf.image.decode_jpeg(image, channels=3)
        feature_img = tf.cast(image, tf.float32)
        relative_img = util.absolute_to_relative_colors(feature_img)
        processed_img = self._preprocess_for_training(relative_img)

        return [processed_img, label]

    def _preprocess_for_training(self, image):
        image = resize_random(image, 256, 480)
        # swapped cropping and flipping because flip needs image shape to be fully defined - should not make a
        # difference
        image = random_crop_to_square(image, 224)
        image = color_noise(image, *self._color_data[2:])
        image = normalize_colors(image, *self._color_data[:2])
        image = random_flip(image)
        return image

    def evaluation_inputs(self):
        # TODO implement
        pass

    def _preprocess_for_evaluation(self, image):
        image = normalize_colors(image, *self._color_data[2:])
        return ten_crop(image)
