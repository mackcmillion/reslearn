#!/bin/env python

import os
import threading

import tensorflow as tf
from datetime import datetime as dt
from tensorflow.python.platform import gfile

from datasets.cifar10 import Cifar10
from datasets.imagenet import ImageNet
from config import FLAGS
from models.resnet34 import ResNet34
from models.resnet6nplus2 import CIFAR10ResNet20, CIFAR10ResNet32, CIFAR10ResNet44, CIFAR10ResNet56, CIFAR10ResNet110, \
    CIFAR10ResNet1202
from train import train
from evaluate import evaluate

MODEL_DICT = {'resnet-34': ResNet34,
              'cifar10-resnet-20': CIFAR10ResNet20,
              'cifar10-resnet-32': CIFAR10ResNet32,
              'cifar10-resnet-44': CIFAR10ResNet44,
              'cifar10-resnet-56': CIFAR10ResNet56,
              'cifar10-resnet-110': CIFAR10ResNet110,
              'cifar10-resnet-1202': CIFAR10ResNet1202
              }
DATASET_DICT = {'cifar10': Cifar10, 'imagenet': ImageNet}


def main(argv=None):  # pylint: disable=unused-argument

    if FLAGS.model not in MODEL_DICT:
        raise ValueError('%s - Unknown model.' % dt.now())
    if FLAGS.dataset not in DATASET_DICT:
        raise ValueError('%s - Unknown dataset' % dt.now())

    dataset = DATASET_DICT[FLAGS.dataset]()
    model = MODEL_DICT[FLAGS.model]()

    if not model.supports_dataset(dataset):
        raise ValueError('%s - %s does not support %s.' % (dt.now(), model.name, dataset.name))

    if not gfile.Exists(FLAGS.checkpoint_path):
        gfile.MkDir(FLAGS.checkpoint_path)
    if not gfile.Exists(FLAGS.summary_path):
        gfile.MkDir(FLAGS.summary_path)

    now = dt.now()
    exp_dirname = FLAGS.experiment_name + ('_%s' % now.strftime('%Y-%m-%d_%H-%M-%S'))
    exp_dirname_eval = exp_dirname + '_test'
    summary_path = os.path.join(FLAGS.summary_path, exp_dirname)
    summary_path_eval = os.path.join(FLAGS.summary_path, exp_dirname_eval)
    checkpoint_path = os.path.join(FLAGS.checkpoint_path, exp_dirname)
    gfile.MkDir(summary_path)
    gfile.MkDir(checkpoint_path)

    dataset.pre_graph()

    training_thread = threading.Thread(target=train,
                                       args=(dataset, model, summary_path, checkpoint_path),
                                       name='training-thread')
    evaluation_thread = threading.Thread(target=evaluate,
                                         args=(dataset, model, summary_path_eval, checkpoint_path),
                                         name='evaluation-thread')

    training_thread.start()
    evaluation_thread.start()

    training_thread.join()
    evaluation_thread.join()

    print '%s - Finished %s.' % (dt.now(), FLAGS.experiment_name)


if __name__ == '__main__':
    tf.app.run()
