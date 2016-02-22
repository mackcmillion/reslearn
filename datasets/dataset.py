from abc import ABCMeta, abstractmethod


class Dataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, num_classes):
        self._num_classes = num_classes

    @abstractmethod
    def pre_graph(self):
        pass

    @abstractmethod
    def preliminary(self):
        pass

    @abstractmethod
    def training_inputs(self):
        pass

    @abstractmethod
    def evaluation_inputs(self):
        pass

    @property
    def num_classes(self):
        return self._num_classes
