import tensorflow as tf
from tensorflow.python.platform import gfile

import metrics
import util
from config import FLAGS
from datasets.dataset import Dataset
from preprocess import resize_random, random_crop_to_square, random_flip, color_noise, normalize_colors, single_crop
from scripts.labelmap import create_label_map_file_yelp
from scripts.meanstddev import compute_overall_mean_stddev


class YelpSmall(Dataset):
    def __init__(self):
        super(YelpSmall, self).__init__('yelp-small', 9)
        self._color_data = None

    def pre_graph(self):
        compute_overall_mean_stddev(overwrite=False, num_threads=FLAGS.num_consuming_threads, num_logs=10)
        self._color_data = util.load_meanstddev(FLAGS.yelp_mean_stddev_path)
        create_label_map_file_yelp(overwrite=False)

    def preliminary(self):
        pass

    def training_inputs(self):
        return self._inputs(FLAGS.yelp_training_set, self._preprocess_for_training)

    def evaluation_inputs(self):
        return self._inputs(FLAGS.yelp_validation_set, self._preprocess_for_evaluation)

    def loss_fn(self, predictions, true_labels):
        return util.mll_error(predictions, true_labels)

    def training_error(self, predictions, true_labels):
        threshold = tf.constant(0.0, dtype=tf.float32, shape=predictions.get_shape())
        thresholded_predictions = tf.greater(predictions, threshold)
        thresholded_predictions = tf.cast(thresholded_predictions, dtype=tf.float32)

        hamming = metrics.hamming_loss(thresholded_predictions, true_labels)

        mean_hamming = tf.reduce_mean(hamming)
        return mean_hamming, 'hamming loss'

    def eval_op(self, predictions, true_labels):
        return metrics.hamming_loss(predictions, true_labels)

    def test_error(self, accumulated, total):
        return accumulated / total, 'hamming loss'

    def _inputs(self, setpath, fn_preprocess):
        fps, labels = self._load_labelmap(setpath)
        filepaths = tf.constant(fps)
        labels = tf.constant(labels, dtype=tf.float32)

        min_num_examples_in_queue = int(FLAGS.min_frac_examples_in_queue * len(fps))

        filename_queue = tf.RandomShuffleQueue(len(fps), min_num_examples_in_queue, [tf.string, tf.float32],
                                               shapes=[[], [self._num_classes]])
        enqueue_op = filename_queue.enqueue_many([filepaths, labels])
        qr = tf.train.QueueRunner(filename_queue, [enqueue_op])
        tf.train.add_queue_runner(qr)

        example_list = [self._read_and_preprocess_image(filename_queue, fn_preprocess) for _ in
                        xrange(FLAGS.num_consuming_threads)]

        image_batch, label_batch = tf.train.shuffle_batch_join(
                example_list,
                batch_size=FLAGS.batch_size,
                capacity=min_num_examples_in_queue + (FLAGS.num_consuming_threads + 2) * FLAGS.batch_size,
                min_after_dequeue=min_num_examples_in_queue,
                shapes=[[32, 32, 3], [self._num_classes]]
        )

        return image_batch, label_batch

    def _load_labelmap(self, filepath):
        if not gfile.Exists(filepath):
            raise ValueError('Label map file not found.')

        filepaths = []
        labels = []

        with open(filepath) as f:
            for line in f:
                fp, lbl = line.split(',')
                lbl_lst = lbl.split(' ')
                lbl_lst = map(int, lbl_lst) if lbl_lst != ['\n'] else []

                # FIXME bp-mll error fn requires empty label sets and full label sets to be excluded
                if lbl_lst and len(lbl_lst) < self._num_classes:
                    filepaths.append(fp)
                    labels.append(lbl_lst)

        return filepaths, util.encode_k_hot_python(labels, self._num_classes)

    @staticmethod
    def _read_and_preprocess_image(filename_queue, fn_preprocess):
        filename, label = filename_queue.dequeue()

        # we want to reuse the filename-label-pair for later training steps
        filename_queue.enqueue([filename, label])

        image = tf.read_file(filename)

        # prepare image
        image = tf.image.decode_jpeg(image, channels=3)
        feature_img = tf.cast(image, tf.float32)
        # relative colors is important for adding color noise
        relative_img = util.absolute_to_relative_colors(feature_img)
        processed_img = fn_preprocess(relative_img)

        return [processed_img, label]

    def _preprocess_for_training(self, image):
        image = resize_random(image, 32, 66)
        # swapped cropping and flipping because flip needs image shape to be fully defined - should not make a
        # difference
        image = random_crop_to_square(image, 32)
        image = color_noise(image, *self._color_data[2:])
        image = normalize_colors(image, *self._color_data[:2])
        image = random_flip(image)
        return image

    # TODO implement better evaluation preprocessing
    def _preprocess_for_evaluation(self, image):
        image = resize_random(image, 32, 66)
        image = single_crop(image, 32)
        image = normalize_colors(image, *self._color_data[:2])
        return image