import tensorflow as tf

from datasets.dataset import Dataset


class Yelp(Dataset):

    def __init__(self):
        super(Yelp, self).__init__('yelp', 9)

    def pre_graph(self):
        pass

    def preliminary(self):
        pass

    def training_inputs(self):
        pass

    def evaluation_inputs(self):
        pass
