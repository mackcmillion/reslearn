from abc import ABCMeta, abstractmethod

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
    def inference(self, x, num_classes):
        pass


class ResidualModel(Model):
    __metaclass__ = ABCMeta

    def __init__(self, name, supported_datasets):
        super(ResidualModel, self).__init__(name, supported_datasets)
        assert FLAGS.adjust_dimensions_strategy in ['A', 'B']
        if FLAGS.adjust_dimensions_strategy == 'A':
            self._adjust_dimensions = 'IDENTITY'
        else:
            self._adjust_dimensions = 'PROJECTION'
