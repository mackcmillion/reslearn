import tensorflow as tf

from datasets.cifar10 import Cifar10
from datasets.imagenet import ImageNet
from hyperparams import FLAGS
from resnet_34 import resnet_34
from train import train

NET_DICT = {'resnet_34': resnet_34}
DATASET_DICT = {'cifar10': Cifar10(), 'imagenet': ImageNet()}


def main(argv=None):  # pylint: disable=unused-argument

    if FLAGS.net not in NET_DICT:
        raise ValueError('Unknown net.')
    if FLAGS.dataset not in DATASET_DICT:
        raise ValueError('Unknown dataset')

    train(DATASET_DICT[FLAGS.dataset], NET_DICT[FLAGS.net])


if __name__ == '__main__':
    tf.app.run()
