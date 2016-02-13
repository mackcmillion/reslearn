import os

from tensorflow.python.platform import gfile

from hyperparams import FLAGS

WNID_LID_MAP = None


def create_label_map_file(overwrite=False, num_logs=10):

    if gfile.Exists(FLAGS.labelmap_path):
        print 'Labelmap file already exists.'
        if overwrite:
            print 'Overwriting file...'
            gfile.Remove(FLAGS.labelmap_path)
        else:
            print 'Nothing to do here.'
            return
        print

    print 'Building filename list...'
    filenames = build_filename_list()

    f = open(FLAGS.labelmap_path, 'w')

    log_mod = len(filenames) / num_logs

    for i, filepath in enumerate(filenames):
        lid = _get_label_id_for_wnid(filepath)
        f.write('%s,%i\n' % (filepath, lid))

        if i != 0 and i % log_mod == 0:
            print '%i/%i of files processed.' % (int(i / log_mod), num_logs)

    f.close()


def build_filename_list():
    image_files = []
    for dirpath, _, filenames in os.walk(FLAGS.train_dir):
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


if __name__ == '__main__':
    create_label_map_file(overwrite=True)
