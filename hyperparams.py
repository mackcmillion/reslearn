import tensorflow as tf

experiment_name = 'resnet_34_test'

# constants specifying training behaviour
tf.app.flags.DEFINE_string('dataset', 'imagenet',
                           """The dataset which to train on.""")

tf.app.flags.DEFINE_string('net', 'resnet_34',
                           """The name of the net to train.""")

# OPTIMIZER = tf.train.AdamOptimizer(learning_rate=0.1)
OPTIMIZER = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)

tf.app.flags.DEFINE_integer('num_classes', 2,
                            """Numbers of classes the input data is divided into.""")

tf.app.flags.DEFINE_integer('training_epochs', 20,
                            """Number of iterations for training.""")

tf.app.flags.DEFINE_integer('batch_size', 5,
                            """Size of the mini-batches used for training.""")

tf.app.flags.DEFINE_float('min_frac_examples_in_queue', 0.01,
                          """The minimum fraction of all examples to be held in the input queue.
                          Ensures good shuffling.""")

tf.app.flags.DEFINE_integer('num_consuming_threads', 3,
                            """Number of threads consuming a filename to produce an image example.""")

# data directory and file paths
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

# target directory and file paths
tf.app.flags.DEFINE_string('summary_path', '/home/max/Studium/Kurse/BA2/%s/summaries' % experiment_name,
                           """Path to save summary files to. Needed for TensorBoard visualization.""")

tf.app.flags.DEFINE_string('checkpoint_path',
                           '/home/max/Studium/Kurse/BA2/%s/checkpoints' % experiment_name,
                           """Path to periodically save checkpoints of the training procedure.""")

FLAGS = tf.app.flags.FLAGS
