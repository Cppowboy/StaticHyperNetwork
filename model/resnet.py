from model.nets.resnet_v2 import resnet_v2_50
import tensorflow as tf
from model.utils import HyperCell


class Resnet50(object):
    def __init__(self, num_classes, hyper_mode=True):
        self.num_classes = num_classes
        self.hyper_mode = hyper_mode

    def build_model(self, batch_images, batch_labels):
        net, endpoints = resnet_v2_50(batch_images, num_classes=self.num_classes, hyper_mode=self.hyper_mode)
        net = tf.squeeze(net)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=batch_labels)
        predictions = endpoints['predictions']
        predictions = tf.squeeze(predictions)
        total_loss = tf.reduce_sum(loss)
        return total_loss, predictions, tf.argmax(predictions, 1)
