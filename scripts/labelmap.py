import random

import tensorflow as tf
import csv
import os

from tensorflow.python.platform import gfile

from config import FLAGS

WNID_LID_MAP = None


def create_label_map_file_yelp_test(overwrite=False, num_logs=10):
    is_test_set = gfile.Exists(FLAGS.yelp_test_set)
    if is_test_set:
        print 'Labelmap file already exists.'
        if overwrite:
            print 'Overwriting files...'
            gfile.Remove(FLAGS.yelp_test_set)
        else:
            print 'Nothing to do here.'
            return
        print

    print 'Building filename list...'
    filenames = build_filename_list(FLAGS.yelp_test_image_path)

    with open(FLAGS.yelp_test_set, 'w') as test_set:
        for filename in filenames:
            photo_id = filename.split('/')[-1]
            if not photo_id.startswith('._'):
                test_set.write('%s,%s\n' % (filename, str(1)))


def create_label_map_file_yelp(overwrite=False, num_logs=10):
    is_training_set = gfile.Exists(FLAGS.yelp_training_set)
    is_validation_set = gfile.Exists(FLAGS.yelp_validation_set)
    if is_training_set and is_validation_set:
        print 'Labelmaps files already exist.'
        if overwrite:
            print 'Overwriting files...'
            gfile.Remove(FLAGS.yelp_training_set)
            gfile.Remove(FLAGS.yelp_validation_set)
        else:
            print 'Nothing to do here.'
            return
        print
    elif is_training_set:
        print 'Training labelmap file already exists. Overwriting...'
        gfile.Remove(FLAGS.yelp_training_set)
    elif is_validation_set:
        print 'Validation labelmap file already exists. Overwriting...'
        gfile.Remove(FLAGS.yelp_validation_set)

    print 'Building filename list...'
    filenames = build_filename_list(FLAGS.yelp_training_image_path)
    photo_to_biz_id = _get_photo_biz_id_map(FLAGS.yelp_training_photo_biz_id_path)
    biz_id_to_labels_train, biz_id_to_labels_validate = _get_biz_id_labels_maps()

    with open(FLAGS.yelp_training_set, 'w') as training_set, open(FLAGS.yelp_validation_set, 'w') as validation_set:
        for filename in filenames:
            photo_id = filename.split('/')[-1]
            if not photo_id.startswith('._'):
                biz_id = photo_to_biz_id[photo_id.split('.')[0]]
                if biz_id in biz_id_to_labels_train:
                    training_set.write('%s,%s\n' % (filename, biz_id_to_labels_train[biz_id]))
                else:
                    validation_set.write('%s,%s\n' % (filename, biz_id_to_labels_validate[biz_id]))


def _get_photo_biz_id_map(path):
    photo_to_biz_id = {}
    with open(path) as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            photo_to_biz_id[row['photo_id']] = row['business_id']
    return photo_to_biz_id


def _get_biz_id_labels_maps():
    biz_id_to_labels = {}
    with open(FLAGS.yelp_biz_id_label_path) as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            biz_id_to_labels[row['business_id']] = row['labels']

    # Randomly split biz_id map into 2/3 training and 1/3 validation set.
    # We have 2000 businesses, so the validation set will have 666 businesses.
    biz_id_to_labels_validate = {}
    for i in xrange(len(biz_id_to_labels) / 3):
        val_biz = random.choice(biz_id_to_labels.keys())
        val_labels = biz_id_to_labels[val_biz]
        del biz_id_to_labels[val_biz]
        biz_id_to_labels_validate[val_biz] = val_labels

    # check if dicts are disjoint
    for key in biz_id_to_labels:
        assert key not in biz_id_to_labels_validate
    for key in biz_id_to_labels_validate:
        assert key not in biz_id_to_labels

    return biz_id_to_labels, biz_id_to_labels_validate


def create_label_map_file(overwrite=False, num_logs=10):
    if gfile.Exists(FLAGS.training_set):
        print 'Labelmap file already exists.'
        if overwrite:
            print 'Overwriting file...'
            gfile.Remove(FLAGS.training_set)
        else:
            print 'Nothing to do here.'
            return
        print

    print 'Building filename list...'
    filenames = build_filename_list(FLAGS.training_images)

    f = open(FLAGS.training_set, 'w')

    log_mod = len(filenames) / num_logs

    for i, filepath in enumerate(filenames):
        lid = _get_label_id_for_wnid(filepath)

        # FIXME remove
        if lid == 508:
            lid = 0
        elif lid == 354:
            lid = 1

        f.write('%s,%i\n' % (filepath, lid))

        if i != 0 and i % log_mod == 0:
            print '%i/%i of files processed.' % (int(i / log_mod), num_logs)

    f.close()


def build_filename_list(*paths):
    image_files = []
    for p in paths:
        for dirpath, _, filenames in os.walk(p):
            image_files += [os.path.join(dirpath, filename) for filename in filenames]
    return image_files


def _get_label_id_for_wnid(filepath):
    global WNID_LID_MAP
    # first access, map not yet loaded
    if not WNID_LID_MAP:
        _load_wnid_lid_map()

    wnid = filepath.split('/')[-1].split('_')[0]
    if wnid not in WNID_LID_MAP:
        raise KeyError('Unknown WNID ' + wnid)
    lid = WNID_LID_MAP[wnid][0]
    # important since labels in ImageNet are 1-based
    return lid - 1


def _load_wnid_lid_map():
    global WNID_LID_MAP
    print 'Loading WNID_LID map...'
    if not gfile.Exists(FLAGS.wnid_lid_path):
        raise ValueError('WNID_LID file not found.')
    WNID_LID_MAP = dict()

    f = open(FLAGS.wnid_lid_path, 'r')
    for line in f:
        contents = line.split(' ')
        WNID_LID_MAP[contents[0]] = (int(contents[1]), contents[2])


def main(argv=None):  # pylint: disable=unused-argument
    create_label_map_file_yelp(overwrite=True)


if __name__ == '__main__':
    tf.app.run()
