from abc import ABCMeta, abstractmethod


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, supported_datasets):
        self._name = name
        self._supported_datasets = supported_datasets

    def supports_dataset(self, dataset):
        return dataset in self._supported_datasets

    @property
    def name(self):
        return self._name

    @abstractmethod
    def inference(self, x, num_classes):
        pass
