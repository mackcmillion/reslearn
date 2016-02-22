import tensorflow as tf

# global run config
tf.app.flags.DEFINE_string('experiment_name', 'resnet_34_test',
                           """Identifier of the experiment run.""")

tf.app.flags.DEFINE_string('dataset', 'cifar10',
                           """The dataset which to train on.""")

tf.app.flags.DEFINE_string('model', 'cifar10-resnet-20',
                           """The name of the net to train.""")

tf.app.flags.DEFINE_string('adjust_dimensions_strategy', 'A',
                           """
                           The method to adjust the dimensions of the input layer when using residual building blocks.
                           Can have the following values:
                           - A Identity mapping is used to match dimensions. Missing filters are padded with zeros.
                           - B Projection mapping is used to match dimensions. Introduces additional parameters which
                               enable residual learning but increase complexity.
                           Option C, using projections even if there is no need to adjust dimensions, is not supported.
                           """)


# constants specifying training and validation behaviour
OPTIMIZER = tf.train.MomentumOptimizer
OPTIMIZER_ARGS = {'momentum': 0.9}

tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                          """The constant float L2 weight decay loss is multiplied with.""")

tf.app.flags.DEFINE_integer('training_epochs', 20,
                            """Number of iterations for training.""")

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Size of the mini-batches used for training.""")

tf.app.flags.DEFINE_float('min_frac_examples_in_queue', 0.4,
                          """The minimum fraction of all examples to be held in the input queue.
                          Ensures good shuffling.""")

# TODO balance consuming threads
tf.app.flags.DEFINE_integer('num_consuming_threads', 3,
                            """Number of threads consuming a filename to produce an image example.""")

tf.app.flags.DEFINE_integer('log_interval', 1,
                            """The number of steps after which to print a log message.""")

tf.app.flags.DEFINE_integer('summary_interval', 1,
                            """The number of steps after which to create a new summary.""")

tf.app.flags.DEFINE_integer('checkpoint_interval', 3,
                            """The number of steps after which to create a new checkpoint.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 10,
                            """Interval seconds in which to poll the checkpoint directory for new checkpoint files.""")

tf.app.flags.DEFINE_integer('max_num_examples', 300,
                            """Maximum number of examples to process in one evaluation run.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_integer('top_k', 1,
                            """When evaluating, compute the top-k-error.""")


# ImageNet data directory and file paths
tf.app.flags.DEFINE_string('training_images', '/home/max/Studium/Kurse/BA2/data/imagenet/synsets',
                           """Directory containing the training image data.""")

tf.app.flags.DEFINE_string('validation_images', '/home/max/Studium/Kurse/BA2/data/imagenet/validation',
                           """Directory containing the validation image data.""")

tf.app.flags.DEFINE_string('wnid_lid_path', '/home/max/Studium/Kurse/BA2/data/imagenet/map_clsloc.txt',
                           """The file where to load the WNID_LID_MAP from.""")

tf.app.flags.DEFINE_string('mean_stddev_path', '/home/max/Studium/Kurse/BA2/data/imagenet/mean_stddev',
                           """Path where to store/load precomputed mean/stddev over a whole dataset.""")

tf.app.flags.DEFINE_string('training_set', '/home/max/Studium/Kurse/BA2/data/imagenet/labelmap',
                           """Path to the file mapping each filename to its label.""")

tf.app.flags.DEFINE_string('validation_set',
                           "/home/max/Studium/Kurse/BA2/data/imagenet/ILSVRC2015_clsloc_ground_truth.txt",
                           """Path to the validation label map.""")

tf.app.flags.DEFINE_string('validation_blacklist',
                           "/home/max/Studium/Kurse/BA2/data/imagenet/ILSVRC2015_clsloc_validation_blacklist.txt",
                           """Path to the validation set blacklist.""")


# CIFAR-10 data directory and file paths
tf.app.flags.DEFINE_string('cifar10_image_path', '/home/max/Studium/Kurse/BA2/data/cifar-10-batches-bin',
                           """Path to training and test images for CIFAR-10 dataset.""")

tf.app.flags.DEFINE_string('cifar10_mean_stddev_path',
                           '/home/max/Studium/Kurse/BA2/data/cifar-10-batches-bin/mean_stddev',
                           """Path where to store/load precomputed mean/stddev over a whole CIFAR-10 dataset.""")


# target directory and file paths
tf.app.flags.DEFINE_string('summary_path', '/home/max/Studium/Kurse/BA2/summaries',
                           """Path to save summary files to. Needed for TensorBoard visualization.""")

tf.app.flags.DEFINE_string('checkpoint_path',
                           '/home/max/Studium/Kurse/BA2/checkpoints',
                           """Path to periodically save checkpoints of the training procedure.""")

FLAGS = tf.app.flags.FLAGS
