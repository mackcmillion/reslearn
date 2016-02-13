import os
import random

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.ops import control_flow_ops as cf
#
# sess = tf.InteractiveSession()
#
# x = tf.range(0, 2 * 4 * 4 * 2)
# x = tf.reshape(x, [2, 4, 4, 2])
#
# print x.eval()
#
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


class CustomWholeFileReader(tf.WholeFileReader):

    def read(self, queue, name=None):
        queue_el = queue.dequeue()
        filename = queue_el[0]
        label = queue_el[1]
        key, value = super(CustomWholeFileReader, self).read(filename)
        return key, value, label


def inputs():

    filename_labels = [['file1', 'label1'], ['file2', 'label2']]
    filename_queue = tf.FIFOQueue(capacity=len(filename_labels), dtypes=tf.string_ref)

    reader = CustomWholeFileReader()
    key, value, label = reader.read(filename_queue)

    label = tf.Print(label, [label])

    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            print sess.run(label)

    except tf.errors.OutOfRangeError:
        print 'All images processed.'
    finally:
        coord.request_stop()

    coord.join(threads)


if __name__ == '__main__':
    inputs()