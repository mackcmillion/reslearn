import tensorflow as tf

from resnet_34 import resnet_34

NET = resnet_34

OPTIMIZER = tf.train.MomentumOptimizer(0.1, 0.9)

tf.app.flags.DEFINE_integer('num_classes', 1000,
                            """Numbers of classes the input data is divided into.""")

tf.app.flags.DEFINE_integer('training_epochs', 100,
                            """Number of iterations for training.""")

tf.app.flags.DEFINE_string('train_dir', '../data/imagenet/synsets',
                           """Directory containing the training data.""")

tf.app.flags.DEFINE_integer('batch_size', 5,
                            """Size of the mini-batches used for training.""")

FLAGS = tf.app.flags.FLAGS
