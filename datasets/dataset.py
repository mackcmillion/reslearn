from abc import ABCMeta, abstractmethod


class Dataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, num_classes):
        self._num_classes = num_classes
        self._name = name

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

    @abstractmethod
    def loss_fn(self, predictions, true_labels):
        pass

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes
