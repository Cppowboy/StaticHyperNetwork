from keras.datasets import cifar10
import numpy as np


class Cifar10(object):
    def __init__(self):
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.y_train = np.eye(self.num_classes)[self.y_train]
        self.y_test = np.eye(self.num_classes)[self.y_test]
