import os

import tensorflow as tf
from datetime import datetime as dt
from tensorflow.python.platform import gfile

from datasets.cifar10 import Cifar10
from datasets.imagenet import ImageNet
from config import FLAGS
from models.resnet34 import ResNet34
from train import train

MODEL_DICT = {'resnet-34': ResNet34}
DATASET_DICT = {'cifar10': Cifar10, 'imagenet': ImageNet}


def main(argv=None):  # pylint: disable=unused-argument

    if FLAGS.model not in MODEL_DICT:
        raise ValueError('Unknown model.')
    if FLAGS.dataset not in DATASET_DICT:
        raise ValueError('Unknown dataset')

    if not gfile.Exists(FLAGS.checkpoint_path):
        gfile.MkDir(FLAGS.checkpoint_path)
    if not gfile.Exists(FLAGS.summary_path):
        gfile.MkDir(FLAGS.summary_path)

    now = dt.now()
    exp_dirname = FLAGS.experiment_name + ('_%s' % now.strftime('%Y-%m-%d_%H-%M-%S'))
    summary_path = os.path.join(FLAGS.summary_path, exp_dirname)
    checkpoint_path = os.path.join(FLAGS.checkpoint_path, exp_dirname)
    gfile.MkDir(summary_path)
    gfile.MkDir(checkpoint_path)

    train(DATASET_DICT[FLAGS.dataset](), MODEL_DICT[FLAGS.model]())


if __name__ == '__main__':
    tf.app.run()
