from abc import ABCMeta, abstractmethod
import tensorflow as tf

from config import FLAGS


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, supported_datasets):
        self._name = name
        self._supported_datasets = supported_datasets

    def supports_dataset(self, dataset):
        return dataset.name in self._supported_datasets

    @property
    def name(self):
        return self._name

    @abstractmethod
    def inference(self, x, num_classes, phase_train):
        pass

    def inference_ten_crop(self, x, num_classes, crop_size, phase_train):
        glimpses = [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        x = tf.transpose(x, [2, 1, 0, 3, 4, 5])
        x, x_flipped = tf.unpack(x)

        all_predictions = []
        for sample in tf.unpack(x) + tf.unpack(x_flipped):
            for glimpse in glimpses:
                offsets = tf.constant(glimpse, dtype=tf.float32, shape=[2])
                offsets = tf.expand_dims(offsets, 0)
                offsets = tf.tile(offsets, [sample.get_shape()[0].value, 1])

                crop = tf.image.extract_glimpse(sample, [crop_size, crop_size], offsets=offsets,
                                                centered=True, normalized=True)
                predictions = self.inference(crop, num_classes, phase_train)
                all_predictions.append(predictions)

        return tf.reduce_mean(tf.pack(all_predictions), reduction_indices=[0])


class ResidualModel(Model):
    __metaclass__ = ABCMeta

    def __init__(self, name, supported_datasets):
        super(ResidualModel, self).__init__(name, supported_datasets)
        assert FLAGS.adjust_dimensions_strategy in ['A', 'B']
        if FLAGS.adjust_dimensions_strategy == 'A':
            self._adjust_dimensions = 'IDENTITY'
        else:
            self._adjust_dimensions = 'PROJECTION'
