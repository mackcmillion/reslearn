from datasets.dataset import Dataset


class Cifar10(Dataset):

    def __init__(self):
        super(Cifar10, self).__init__(1000)

    def preliminary(self):
        pass

    def training_inputs(self):
        pass
