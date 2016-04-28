import csv
import json
import tensorflow as tf

from datasets.yelp import Yelp

IMAGE_PREDICTION_PATH = '/home/max/Studium/Kurse/BA2/results/prediction_map_val'
PHOTO_BIZ_ID_PATH = '/home/max/Studium/Kurse/BA2/data/yelp/train_photo_to_biz_ids.csv'
TARGET_FILE = '/home/max/Studium/Kurse/BA2/results/submission.csv'

CLASSIFICATION_THRESHOLD = 0.5


def hamming_loss(sess, true_labels, predictions):
    pred = tf.placeholder(dtype=tf.float32, shape=[9])
    lbls = tf.placeholder(dtype=tf.float32, shape=[9])
    pred = tf.expand_dims(pred, 0)
    lbls = tf.expand_dims(lbls, 0)
    eval_op = Yelp().eval_op(pred, lbls)
    return sess.run(eval_op, feed_dict={pred: predictions, lbls: true_labels})[0]


def hamming_loss_map(img_lbl_pred_map):
    sess = tf.Session()
    hloss_map = {}
    for image in img_lbl_pred_map:
        print 'Computing hamming loss for ' + image
        true_labels, predictions = img_lbl_pred_map[image]
        hloss = hamming_loss(sess, true_labels, predictions)
        hloss_map[image] = hloss
    return hloss_map


def total_hamming_loss(img_lbl_pred_map):
    hloss_map = hamming_loss_map(img_lbl_pred_map)
    return sum(hloss_map[image] for image in hloss_map) / len(hloss_map)


def accumulate_for_biz(img_lbl_pred_map):
    accum = {}
    img_biz_id_map = _get_photo_biz_id_map()
    for image in img_lbl_pred_map:
        img_id = image.split('/')[-1].split('.')[0]
        biz_id = img_biz_id_map[img_id]
        if biz_id in accum:
            accum[biz_id][1].append(img_lbl_pred_map[image][1])
        else:
            accum[biz_id] = (img_lbl_pred_map[image][0], [img_lbl_pred_map[image][1]])

    new_map = {}
    for biz in accum:
        image_scores = accum[biz][1]
        zipped = zip(*image_scores)
        avg_scores = []
        for single_label_pred in zipped:
            avg_scores.append(sum(single_label_pred) / len(single_label_pred))
        new_map[biz] = (accum[biz][0], avg_scores)

    return new_map


def _get_photo_biz_id_map():
    photo_to_biz_id = {}
    with open(PHOTO_BIZ_ID_PATH) as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            photo_to_biz_id[row['photo_id']] = row['business_id']
    return photo_to_biz_id


def read_prediction_file(path):
    img_lbl_pred_map = {}
    with open(path, 'r') as f:
        for line in f:
            split = line.split(',')
            image = split[0]
            true_labels = ','.join(split[1:10])
            predictions = ','.join(split[10:])
            true_labels = json.loads(true_labels)
            predictions = json.loads(predictions)
            img_lbl_pred_map[image] = (true_labels, predictions)

    return img_lbl_pred_map


def make_prediction(img_lbl_pred_map):
    sess = tf.Session()
    final = 'business_id,labels\n'
    for image in img_lbl_pred_map:
        print 'Predicting ' + image
        preds = img_lbl_pred_map[image][1]
        preds = predict(sess, preds)

        indices = []
        for i, pred in enumerate(preds):
            if pred:
                indices.append(str(i))

        final += image + ',' + ' '.join(indices) + '\n'

    with open(TARGET_FILE, 'w') as f:
        f.write(final)

    print 'Done.'


def predict(sess, predictions):
    predictions = tf.sigmoid(predictions)
    threshold = tf.constant(CLASSIFICATION_THRESHOLD, dtype=tf.float32, shape=predictions.get_shape())
    thresholded_predictions = tf.greater(predictions, threshold)
    return sess.run(thresholded_predictions)


if __name__ == '__main__':
    ilpm = read_prediction_file(IMAGE_PREDICTION_PATH)
    acc_ilpm = accumulate_for_biz(ilpm)
    make_prediction(acc_ilpm)
