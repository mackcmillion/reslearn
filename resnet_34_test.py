import tensorflow as tf

from resnet_34 import resnet_34


def main():
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    sess = tf.Session()
    net = resnet_34(x)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('summaries/resnet_34_test', sess.graph_def)
    sess.run(tf.initialize_all_variables())

if __name__ == '__main__':
    main()