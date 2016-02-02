import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.range(0, 2 * 4 * 4 * 2)
x = tf.reshape(x, [2, 4, 4, 2])

print x.eval()

x = tf.reshape(x, [-1, 4])

print
print x.eval()

x_list = tf.unpack(x)

print
print x_list

x_packed = tf.pack(x_list[0::2])

print
print x_packed.eval()

x = tf.reshape(x_packed, [-1])
x_list = tf.unpack(x)
x_packed = tf.pack(x_list[0::2])
x = tf.reshape(x_packed, [-1, 2, 2, 2])

print
print x.eval()
