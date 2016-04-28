import csv

import tensorflow as tf

# image = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
# print image.eval()
# remainders = tf.transpose(tf.tile(tf.expand_dims(image, 0), [3, 1, 1]), [1, 2, 0])
# remainders = tf.constant([[[1, 2, 3]], [[4, 5, 6]]])
# print remainders.get_shape()
# print remainders.eval()
# remainders_stack = tf.pack([remainders, remainders, remainders])
# remainders_stack_transposed = tf.transpose(remainders_stack, [3, 1, 2, 0])
# pseudo_squares = tf.mul(remainders_stack, remainders_stack_transposed)
# print pseudo_squares.get_shape()
# print pseudo_squares.eval()
# sum_of_squares = tf.reduce_sum(pseudo_squares, [1, 2])
#
# print sum_of_squares.eval()

#
# def test_mean_stddev():
#     _compute_overall_mean_stddev(overwrite=True,
#                                  num_threads=4,
#                                  num_logs=10,
#                                  image_op=_mock_images,
#                                  filenames=[],
#                                  mean_stddev_path=FLAGS.cifar10_mean_stddev_path,
#                                  relative_colors=False,
#                                  num_files=1000)
#
#
# def _mock_images(filenames, relative_colors):
#     return tf.random_normal([32, 32, 3], dtype=tf.float32, mean=0.0, stddev=1.0)
#
#
# if __name__ == '__main__':
#     test_mean_stddev()

# def test_conv_simulation():
#     sess = tf.InteractiveSession()
#     # x = tf.constant([[[1, 2], [1, 2]], [[1, 2], [1, 2]]], dtype=tf.float32, shape=[2, 2, 2])
#     # x = tf.expand_dims(x, 0)
#     # print x.get_shape()
#     #
#     w = tf.constant([[1, 1], [1, 1]], dtype=tf.float32, shape=[2, 2])
#     #
#     # result = _convolve(x, w)
#     # print result.get_shape()
#     # print result.eval()
#     #
#     # print
#
#     x = tf.constant([[[1, 2], [1, 2], [1, 2], [1, 2]],
#                      [[1, 2], [1, 2], [1, 2], [1, 2]],
#                      [[1, 2], [1, 2], [1, 2], [1, 2]],
#                      [[1, 2], [1, 2], [1, 2], [1, 2]]], dtype=tf.float32, shape=[4, 4, 2])
#     x = tf.expand_dims(x, 0)
#     # print x.get_shape()
#     # print x.eval()
#
#     w = tf.expand_dims(tf.expand_dims(w, 0), 0)
#     extracted = tf.nn.max_pool(_mask_input(x), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#     conv = tf.nn.conv2d(extracted, w, [1, 1, 1, 1], padding='SAME')
#     print conv.eval()
#
#     w = tf.constant([[1, 1], [1, 1]], dtype=tf.float32, shape=[2, 2])
#
#     result = _convolve(x, w)
#     print result.get_shape()
#     print result.eval()
#
#
# if __name__ == '__main__':
#     test_conv_simulation()

# sess = tf.InteractiveSession()
#
# x = tf.range(0, 2 * 4 * 4 * 2)
# x = tf.reshape(x, [2, 4, 4, 2])
# x = tf.cast(x, tf.float32)
#
# print x.eval()
#
# x_shape = x.get_shape().as_list()
# mask = [[row % 2 == 0 and column % 2 == 0 for column in xrange(x_shape[2])] for row in xrange(x_shape[1])]
# mask = tf.cast(tf.constant(mask, dtype=tf.bool), tf.float32)
#
#
# mask = tf.expand_dims(tf.expand_dims(mask, 0), 3)
# mask = tf.tile(mask, [x_shape[0], 1, 1, x_shape[3]])
# print mask
# print mask.eval()
#
# x_masked = tf.mul(x, mask)
# print x_masked.eval()
#
# print tf.nn.max_pool(x_masked, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID').eval()
#
# padded = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 2]])
#
# print
# print padded.eval()

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

# sess = tf.InteractiveSession()
#
# predictions = tf.constant([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
# true_labels = tf.constant([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=tf.float32)


# import tensorflow as tf
#
# a = tf.constant([27, 35, 8], dtype=tf.int32, shape=[3], name='a')
# b = tf.constant([15, 7, 34], dtype=tf.int32, shape=[3], name='b')
#
# c = tf.add(a, b)
#
#
# sess = tf.Session()
# writer = tf.train.SummaryWriter('/home/max/Studium/Kurse/BA2/tex/summaries', sess.graph_def)
# result = sess.run(c)
# print c
# print result
# sess.close()

# predictions = tf.cast(predictions, tf.bool)
# true_labels = tf.cast(true_labels, tf.bool)
#
# misclassified = tf.logical_xor(predictions, true_labels)
# hamming_loss = tf.reduce_mean(tf.cast(misclassified, tf.float32), reduction_indices=1)
# print hamming_loss.eval()



#
# true_positives = tf.equal(2.0, predictions + true_labels)
# true_positives = tf.reduce_sum(tf.cast(true_positives, tf.float32), reduction_indices=1)
#
# predicted_positives = tf.reduce_sum(predictions, reduction_indices=1)
# actual_positives = tf.reduce_sum(true_labels, reduction_indices=1)
#
# precision = true_positives / predicted_positives
# recall = true_positives / actual_positives
#
# f1 = 2 * (precision * recall) / (precision + recall)
#
# print precision.eval()
# print recall.eval()

# sess = tf.InteractiveSession()
#
# images = tf.ones([2, 4, 4, 1], dtype=tf.float32)
#
# glimpse = tf.image.extract_glimpse(images, [2, 2], [[0.0, 0.0], [0.0, 0.0]], centered=True, normalized=True)
# print glimpse
# print glimpse.eval()

# PATH = "/home/max/Studium/Kurse/BA2/data/yelp/test_photo_to_biz.csv"
# photo_to_biz_id = {}
#
# with open(PATH) as f:
#     csvreader = csv.DictReader(f)
#     for row in csvreader:
#         photo_to_biz_id[row['business_id']] = row['photo_id']
#
# print len(photo_to_biz_id)
