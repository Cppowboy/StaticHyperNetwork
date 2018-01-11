from keras.datasets import mnist
import numpy as np


class Mnist(object):
    def __init__(self):
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = np.expand_dims(self.x_train, axis=3)
        self.x_test = np.expand_dims(self.x_test, axis=3)
        self.y_train = np.eye(self.num_classes)[self.y_train]
        self.y_test = np.eye(self.num_classes)[self.y_test]
