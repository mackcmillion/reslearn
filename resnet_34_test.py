import tensorflow as tf

from hyperparams import FLAGS
from resnet_34 import resnet_34
from train import train

NET_DICT = {'resnet_34': resnet_34}


def main(argv=None):  # pylint: disable=unused-argument

    if FLAGS.net not in NET_DICT:
        raise ValueError('Unknown net.')

    train(NET_DICT[FLAGS.net])


if __name__ == '__main__':
    tf.app.run()
