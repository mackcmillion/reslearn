#!/bin/env python

import os
import threading
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.python.platform import gfile

from datasets.cifar10 import Cifar10
from datasets.imagenet import ImageNet
from datasets.yelp_small import YelpSmall
from evaluate import evaluate
from models.resnet18 import ResNet18
from config import FLAGS
from datasets.yelp import Yelp
from models.resnet34 import ResNet34
from models.resnet6nplus2 import CIFAR10ResNet20, CIFAR10ResNet32, CIFAR10ResNet44, CIFAR10ResNet56, CIFAR10ResNet110, \
    CIFAR10ResNet1202
from train import train
from util import DATE_FORMAT

MODEL_DICT = {'resnet-18': ResNet18,
              'resnet-34': ResNet34,
              'cifar10-resnet-20': CIFAR10ResNet20,
              'cifar10-resnet-32': CIFAR10ResNet32,
              'cifar10-resnet-44': CIFAR10ResNet44,
              'cifar10-resnet-56': CIFAR10ResNet56,
              'cifar10-resnet-110': CIFAR10ResNet110,
              'cifar10-resnet-1202': CIFAR10ResNet1202
              }
DATASET_DICT = {'cifar10': Cifar10, 'imagenet': ImageNet, 'yelp': Yelp, 'yelp-small': YelpSmall}


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

    no_dirname = True
    exp_dirname = None
    if FLAGS.resume or (FLAGS.eval and not FLAGS.train):
        try:
            exp_dirname = _get_latest_dir()
            no_dirname = False
        except ValueError as e:
            print e
            print '%s - Could not find directory to resume. Starting fresh run of %s.' \
                  % (dt.now(), FLAGS.experiment_name)

    if no_dirname:
        now = dt.now()
        exp_dirname = FLAGS.experiment_name + ('_%s' % now.strftime(DATE_FORMAT))

    exp_dirname_eval = exp_dirname + '_test'
    summary_path = os.path.join(FLAGS.summary_path, exp_dirname)
    summary_path_eval = os.path.join(FLAGS.summary_path, exp_dirname_eval)
    checkpoint_path = os.path.join(FLAGS.checkpoint_path, exp_dirname)

    if not gfile.Exists(summary_path):
        gfile.MkDir(summary_path)
    if not gfile.Exists(checkpoint_path):
        gfile.MkDir(checkpoint_path)

    if gfile.Exists(FLAGS.learning_rate_file_path):
        gfile.Remove(FLAGS.learning_rate_file_path)
    with open(FLAGS.learning_rate_file_path, 'w') as lrfile:
        lrfile.write(str(FLAGS.initial_learning_rate))

    dataset.pre_graph()

    training_thread = None
    evaluation_thread = None
    if FLAGS.train:
        training_thread = threading.Thread(target=train,
                                           args=(dataset, model, summary_path, checkpoint_path),
                                           name='training-thread')
    if FLAGS.eval:
        evaluation_thread = threading.Thread(target=evaluate,
                                             args=(dataset, model, summary_path_eval, checkpoint_path),
                                             name='evaluation-thread')

    if FLAGS.train:
        training_thread.start()
    if FLAGS.eval:
        evaluation_thread.start()

    if FLAGS.train:
        training_thread.join()
    if FLAGS.eval:
        evaluation_thread.join()

    print '%s - Finished %s.' % (dt.now(), FLAGS.experiment_name)


def _get_latest_dir():
    dirs = {}
    for f in gfile.ListDirectory(FLAGS.checkpoint_path):
        if f.startswith(FLAGS.experiment_name):
            split = f.split('_')
            key = dt.strptime('_'.join(split[-2:]), DATE_FORMAT)
            dirs[dt.strptime('_'.join(split[-2:]), DATE_FORMAT)] = f
    if dirs == {}:
        raise ValueError
    maxdir = dirs[max(dirs)]
    return maxdir


if __name__ == '__main__':
    tf.app.run()
