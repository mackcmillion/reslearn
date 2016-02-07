import tensorflow as tf


tf.app.flags.DEFINE_integer('training_steps', 100,
                            """Number oof iterations for training.""")

tf.app.flags.DEFINE_string('train_dir', '../data/imagenet/synset_sports',
                           """Directory containing the training data.""")

tf.app.flags.DEFINE_integer('batch_size', 5,
                            """Size of the mini-batches used for training.""")

FLAGS = tf.app.flags.FLAGS














