import tensorflow as tf
from model.utils import HyperCell


class SimpleCNN(object):
    def __init__(self, num_classes=10, f_size=7, in_size=16, out_size=16, batch_size=64, hyper_mode=True,
                 conv_weight_initializer=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        self.num_classes = num_classes
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.hyper_mode = hyper_mode
        if kernel_initializer is None:
            self.kernel_intializer = tf.orthogonal_initializer(1.0)
        else:
            self.kernel_intializer = kernel_initializer
        if bias_initializer is None:
            self.bias_initializer = tf.constant_initializer(0.0)
        else:
            self.bias_initializer = bias_initializer

    def build_model(self, batch_images, batch_labels):
        f_size = self.f_size
        in_size = self.in_size
        out_size = self.out_size
        conv1_weights = tf.Variable(tf.truncated_normal([f_size, f_size, 1, out_size], stddev=0.01),
                                    name="conv1_weights")
        conv1_biases = tf.Variable(tf.zeros([in_size]), name="conv1_biases")

        net = tf.nn.conv2d(batch_images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net + conv1_biases)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        if self.hyper_mode:
            hypercell = HyperCell(self.f_size, self.in_size, self.out_size, 4, name='hypercell')
            net = hypercell.conv2d(net, out_size, kernel_size=f_size, stride=[1, 1, 1, 1], padding='SAME',
                                   scope='conv2')
        else:
            conv2_weights = tf.Variable(tf.truncated_normal([f_size, f_size, in_size, out_size], stddev=0.01),
                                        name="conv2_weights")
            conv2_biases = tf.Variable(tf.zeros([out_size]), name="conv2_biases")
            net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_biases
        net = tf.nn.relu(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Reshapes the hidden units such that instead of 2D maps, they are 1D vectors:
        net = tf.layers.flatten(inputs=net)
        logits = tf.layers.dense(inputs=net, units=self.num_classes, use_bias=True,
                                 kernel_initializer=self.kernel_intializer, bias_initializer=self.bias_initializer)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)
        # Specify the loss function:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
        total_loss = tf.reduce_sum(cross_entropy)
        return total_loss, probabilities, predictions
