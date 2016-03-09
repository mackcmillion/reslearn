import tensorflow as tf

import util
from config import FLAGS
from datasets.dataset import Dataset
from scripts.labelmap import create_label_map_file, create_label_map_file_yelp
from scripts.meanstddev import compute_overall_mean_stddev


class Yelp(Dataset):

    def __init__(self):
        super(Yelp, self).__init__('yelp', 9)
        self._color_data = None

    def pre_graph(self):
        compute_overall_mean_stddev(overwrite=False, num_threads=FLAGS.num_consuming_threads, num_logs=10)
        self._color_data = util.load_meanstddev(FLAGS.yelp_mean_stddev_path)
        create_label_map_file_yelp(overwrite=False)

    def preliminary(self):
        pass

    def training_inputs(self):
        pass

    def evaluation_inputs(self):
        pass
