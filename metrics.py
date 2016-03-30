# this file contains multiple evaluation metrics for single- and multi-label classification
import tensorflow as tf

from config import FLAGS


def in_top_k(predictions, true_labels):
    # softmax is not necessary here
    # predictions = tf.nn.softmax(predictions, name='eval_softmax')
    return tf.nn.in_top_k(predictions, true_labels, FLAGS.top_k)


def f1_score(predictions, true_labels):
    precision, recall = _compute_precision_and_recall(predictions, true_labels)
    return 2 * (precision * recall) / (precision + recall)


def _compute_precision_and_recall(predictions, true_labels):
    true_positives = tf.equal(2.0, predictions + true_labels)
    true_positives = tf.reduce_sum(tf.cast(true_positives, tf.float32), reduction_indices=1)

    predicted_positives = tf.reduce_sum(predictions, reduction_indices=1)
    actual_positives = tf.reduce_sum(true_labels, reduction_indices=1)

    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    return precision, recall


def hamming_loss(predictions, true_labels):
    return tf.reduce_mean(_hamming_distance(predictions, true_labels), reduction_indices=1)


def _hamming_distance(predictions, true_labels):
    predictions = tf.cast(predictions, tf.bool)
    true_labels = tf.cast(true_labels, tf.bool)

    return tf.cast(tf.logical_xor(predictions, true_labels), tf.float32)
