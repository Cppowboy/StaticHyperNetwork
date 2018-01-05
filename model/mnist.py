import tensorflow as tf
import numpy as np
from model.utils import super_linear


class MNIST(object):
    def __init__(self, hps_model, reuse=False, gpu_mode=True, is_training=True):
        self.is_training = is_training
        with tf.variable_scope('conv_mnist', reuse=reuse):
            if not gpu_mode:
                with tf.device("/cpu:0"):
                    print "model using cpu"
                    self.build_model(hps_model)
            else:
                self.build_model(hps_model)

    def build_model(self, hps_model):

        self.hps = hps_model

        self.model_path = self.hps.model_path
        self.model_save_path = self.model_path + 'mnist'

        self.batch_images = tf.placeholder(tf.float32,
                                           [self.hps.batch_size, self.hps.x_dim, self.hps.x_dim, self.hps.c_dim])
        self.batch_labels = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.num_classes])  # one-hot labels.

        '''
        settings for architecture:
        '''
        f_size = 7
        in_size = 16
        out_size = 16
        z_dim = 4

        conv1_weights = tf.Variable(tf.truncated_normal([f_size, f_size, 1, out_size], stddev=0.01),
                                    name="conv1_weights")

        if self.hps.hyper_mode:
            # the static hypernetwork is inside this if statement.
            w1 = tf.get_variable('w1', [z_dim, out_size * f_size * f_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
            b1 = tf.get_variable('b1', [out_size * f_size * f_size], initializer=tf.constant_initializer(0.0))
            z2 = tf.get_variable("z_signal_2", [1, z_dim], tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.01))
            w2 = tf.get_variable('w2', [z_dim, in_size * z_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
            b2 = tf.get_variable('b2', [in_size * z_dim], initializer=tf.constant_initializer(0.0))
            h_in = tf.matmul(z2, w2) + b2
            h_in = tf.reshape(h_in, [in_size, z_dim])
            h_final = tf.matmul(h_in, w1) + b1
            kernel2 = tf.reshape(h_final, (out_size, in_size, f_size, f_size))
            conv2_weights = tf.transpose(kernel2)
        else:
            conv2_weights = tf.Variable(tf.truncated_normal([f_size, f_size, in_size, out_size], stddev=0.01),
                                        name="conv2_weights")

        self.conv1_weights = conv1_weights
        self.conv2_weights = conv2_weights

        conv1_biases = tf.Variable(tf.zeros([in_size]), name="conv1_biases")

        net = tf.nn.conv2d(self.batch_images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net + conv1_biases)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2_biases = tf.Variable(tf.zeros([out_size]), name="conv2_biases")

        net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net + conv2_biases)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Reshapes the hidden units such that instead of 2D maps, they are 1D vectors:
        net = tf.reshape(net, [self.hps.batch_size, -1])

        net = super_linear(net, self.hps.num_classes, scope='fc_final')

        self.logits = net
        self.probabilities = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, 1)

        # Specify the loss function:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.batch_labels)
        self.loss = tf.reduce_mean(cross_entropy)
        # tf.scalar_summary('Total Loss', self.loss)

        # Specify the optimization scheme:
        self.lr = tf.Variable(self.hps.lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          self.hps.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # model saver
        self.saver = tf.train.Saver(tf.all_variables())

    def update_lr(self, sess):
        lr = sess.run(self.lr)
        lr *= self.hps.lr_decay
        sess.run(tf.assign(self.lr, np.maximum(lr, self.hps.min_lr)))

    def partial_train(self, sess, batch_images, batch_labels):
        _, loss, pred, lr = sess.run((self.train_op, self.loss, self.predictions, self.lr),
                                     feed_dict={self.batch_images: batch_images, self.batch_labels: batch_labels})
        return loss, pred, lr

    def partial_eval(self, sess, batch_images, batch_labels):
        loss, pred = sess.run((self.loss, self.predictions),
                              feed_dict={self.batch_images: batch_images, self.batch_labels: batch_labels})
        return loss, pred

    def save_model(self, sess, epoch=0):
        checkpoint_path = self.model_save_path
        print "saving model: ", checkpoint_path
        self.saver.save(sess, checkpoint_path, global_step=epoch)

    def load_model(self, sess):
        checkpoint_path = self.model_path
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print "loading model: ", ckpt.model_checkpoint_path
        self.saver.restore(sess, ckpt.model_checkpoint_path)
