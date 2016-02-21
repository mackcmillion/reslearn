import tensorflow as tf

from datasets.cifar10 import Cifar10
from datasets.imagenet import ImageNet
from hyperparams import FLAGS
from models.resnet34 import ResNet34
from train import train

MODEL_DICT = {'resnet-34': ResNet34}
DATASET_DICT = {'cifar10': Cifar10, 'imagenet': ImageNet}


def main(argv=None):  # pylint: disable=unused-argument

    if FLAGS.model not in MODEL_DICT:
        raise ValueError('Unknown model.')
    if FLAGS.dataset not in DATASET_DICT:
        raise ValueError('Unknown dataset')

    train(DATASET_DICT[FLAGS.dataset](), MODEL_DICT[FLAGS.model]())


if __name__ == '__main__':
    tf.app.run()
