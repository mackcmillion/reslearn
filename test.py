import os
import random

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.ops import control_flow_ops as cf
#
sess = tf.InteractiveSession()

x = tf.range(0, 2 * 4 * 4 * 2)
x = tf.reshape(x, [2, 4, 4, 2])
x = tf.cast(x, tf.float32)

print x.eval()

x_shape = x.get_shape().as_list()
mask = [[row % 2 == 0 and column % 2 == 0 for column in xrange(x_shape[2])] for row in xrange(x_shape[1])]
mask = tf.cast(tf.constant(mask, dtype=tf.bool), tf.float32)


mask = tf.expand_dims(tf.expand_dims(mask, 0), 3)
mask = tf.tile(mask, [x_shape[0], 1, 1, x_shape[3]])
print mask
print mask.eval()

x_masked = tf.mul(x, mask)
print x_masked.eval()

print tf.nn.max_pool(x_masked, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME').eval()

# x = tf.reshape(x, [-1, 4])
#
# print
# print x.eval()
#
# x_list = tf.unpack(x)
#
# print
# print x_list
#
# x_packed = tf.pack(x_list[0::2])
#
# print
# print x_packed.eval()
#
# x = tf.reshape(x_packed, [-1])
# x_list = tf.unpack(x)
# x_packed = tf.pack(x_list[0::2])
# x = tf.reshape(x_packed, [-1, 2, 2, 2])
#
# print
# print x.eval()

# from hyperparams import FLAGS
#
# sess = tf.Session()
# image = gfile.FastGFile(tf.constant(os.path.join(FLAGS.train_dir, 'n00523513_150.JPEG')), 'rb').read()
# # image = tf.image.decode_jpeg(os.path.join(FLAGS.train_dir, 'n00523513_150.JPEG'))
# image_tensor = tf.image.decode_jpeg(image)
# print image_tensor
# shape = tf.shape(image_tensor)
# s = sess.run(shape)
# height = s[0]
# width = s[1]
# new_shorter_edge = random.choice(xrange(256, 480 + 1))
# if height <= width:
#     new_height = new_shorter_edge
#     new_width = (width / height) * new_shorter_edge
# else:
#     new_width = new_shorter_edge
#     new_height = (height / width) * new_shorter_edge
#
# print height, new_height
#
# image_tensor = tf.image.resize_images(image_tensor, new_height, new_width)
# image_tensor.set_shape([new_height, new_width, s[2]])
# image = tf.image.random_flip_left_right(image_tensor)
# print s
#
# from hyperparams import FLAGS
#
#
# class CustomWholeFileReader(tf.WholeFileReader):
#
#     def read(self, queue, name=None):
#         queue_el = queue.dequeue()
#         filename = queue_el[0]
#         label = queue_el[1]
#         key, value = super(CustomWholeFileReader, self).read(filename)
#         return key, value, label
#
#
# def inputs():
#
#     filename_labels = [['file1', 'label1'], ['file2', 'label2']]
#     filename_queue = tf.FIFOQueue(capacity=len(filename_labels), dtypes=tf.string_ref)
#
#     reader = CustomWholeFileReader()
#     key, value, label = reader.read(filename_queue)
#
#     label = tf.Print(label, [label])
#
#     sess = tf.Session()
#     coord = tf.train.Coordinator()
#     sess.run(tf.initialize_all_variables())
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     try:
#         while not coord.should_stop():
#             print sess.run(label)
#
#     except tf.errors.OutOfRangeError:
#         print 'All images processed.'
#     finally:
#         coord.request_stop()
#
#     coord.join(threads)
#
#
# EIGVECS = None
# EIGVALS = None
#
#
# def _color_noise(image):
#     total_pixels = tf.shape(image)[0] * tf.shape(image)[1]
#     multiples = tf.pack([224, 224, 1])
#
#     alpha = tf.random_normal([3], 0.0, 0.1, dtype=tf.float32)
#     q = tf.matmul(EIGVECS, tf.expand_dims(alpha * EIGVALS, 1))
#     q = tf.squeeze(q)
#     q = tf.expand_dims(tf.expand_dims(q, 0), 0)
#     q = tf.Print(q, [q], summarize=1000)
#     q = tf.tile(q, [2, 2, 1])
#     # q = tf.pack([q for _ in xrange(2)])
#     # q = tf.pack([q for _ in xrange(2)])
#     print q
#     # q = tf.reshape(tf.tile(q, tf.expand_dims(total_pixels, 0)), reshape_to)
#     q = tf.Print(q, [q], summarize=1000)
#
#     return image + q
#
#
# def _load_meanstddev():
#     global MEAN, STDDEV, EIGVALS, EIGVECS
#     # load precomputed mean/stddev
#     if not gfile.Exists(FLAGS.mean_stddev_path):
#         # print 'Mean/stddev file not found. Computing. This might potentially take a long time...'
#         raise ValueError('Mean/stddev file not found.')
#
#     assert gfile.Exists(FLAGS.mean_stddev_path)
#     mean_stddev_string = open(FLAGS.mean_stddev_path, 'r').read().split('\n')
#     mean_str = mean_stddev_string[0][1:-1].split(',')
#     stddev_str = mean_stddev_string[1][1:-1].split(',')
#     eigval_str = mean_stddev_string[2][1:-1].split(',')
#     eigvecs_str = mean_stddev_string[3][1:-1].split(' ')
#
#     MEAN = tf.constant([float(mean_str[0]), float(mean_str[1]), float(mean_str[2])], dtype=tf.float32)
#     STDDEV = tf.constant([float(stddev_str[0]), float(stddev_str[1]), float(stddev_str[2])], dtype=tf.float32)
#     EIGVALS = tf.constant([float(eigval_str[0]), float(eigval_str[1]), float(eigval_str[2])], dtype=tf.float32)
#     eigvecs = []
#     for eigvec_str in eigvecs_str:
#         print eigvec_str
#         eigvec = eigvec_str[1:-1].split(',')
#         print eigvec
#         eigvecs.append([float(eigvec[0]), float(eigvec[1]), float(eigvec[2])])
#     EIGVECS = tf.constant(eigvecs, dtype=tf.float32, shape=[3, 3])
#
#
# if __name__ == '__main__':
#     image = tf.constant([[[0, 0, 0], [0, 0, 0]],
#                           [[0, 0, 0], [0, 0, 0]]], dtype=tf.float32)
#     print image.get_shape()
#
#     _load_meanstddev()
#
#     sess = tf.Session()
#     sess.run(tf.initialize_all_variables())
#     print sess.run(_color_noise(image))