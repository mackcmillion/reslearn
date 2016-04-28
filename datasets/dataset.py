from abc import ABCMeta, abstractmethod


class Dataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, num_classes, num_training_images, num_validation_images):
        self._num_classes = num_classes
        self._name = name
        self._num_training_images = num_training_images
        self._num_validation_images = num_validation_images

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

    @abstractmethod
    def training_error(self, predictions, true_labels):
        pass

    @abstractmethod
    def eval_op(self, predictions, true_labels):
        pass

    @abstractmethod
    def test_error(self, accumulated, total):
        pass

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_training_images(self):
        return self._num_training_images

    @property
    def num_evaluation_images(self):
        return self._num_validation_images
