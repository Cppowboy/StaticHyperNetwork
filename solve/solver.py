import tensorflow as tf
from datasets import Mnist, Cifar10
from model import SimpleCNN
from math import ceil
import numpy as np
from model.utils import ConvWeight


class Solver(object):
    def __init__(self, dataset='mnist', model='simplecnn', **kwargs):
        if dataset == 'mnist':
            self.dataset = Mnist()
        elif dataset == 'cifar10':
            self.dataset = Cifar10()
        else:
            raise NotImplementedError
        conv_weight_initializer = ConvWeight()
        if model == 'simplecnn':
            self.model = SimpleCNN(conv_weight_initializer=conv_weight_initializer, hyper_mode=True)
        else:
            raise NotImplementedError
        self.x_dim = kwargs.pop('x_dim', 28)
        self.c_dim = kwargs.pop('c_dim', 1)
        self.num_classes = kwargs.pop('num_classes', 10)
        self.batch_size = kwargs.pop('batch_size', 1024)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.learning_rate = kwargs.pop('learning_rate', 0.0005)
        self.lr_decay = kwargs.pop('lr_decay', 0.99)
        self.grad_clip = kwargs.pop('grad_clip', 100.0)
        self.optimize_method = kwargs.pop('optimizer', 'adam')

    def train(self):
        x_train, y_train = self.dataset.x_train, self.dataset.y_train
        x_test, y_test = self.dataset.x_test, self.dataset.y_test

        batch_images = tf.placeholder(dtype=tf.float32, shape=[None, self.x_dim, self.x_dim, self.c_dim])
        batch_labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes])
        loss_op, prob_op, predict_op = self.model.build_model(batch_images, batch_labels)

        n_samples = len(x_train)
        n_iterations = int(ceil(n_samples / float(self.batch_size)))
        # learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.learning_rate, global_step=global_step, decay_steps=n_iterations,
                                        decay_rate=self.lr_decay)
        if self.optimize_method == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
        else:
            raise NotImplementedError
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars), self.grad_clip)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        for epoch in range(self.max_epoch):
            # shuffle data
            permutation = np.random.permutation(n_samples)
            x_train = x_train[permutation]
            y_train = y_train[permutation]
            for i in range(n_iterations):
                batch_x = x_train[i * self.batch_size:(i + 1) * self.batch_size]
                batch_y = y_train[i * self.batch_size:(i + 1) * self.batch_size]
                _, loss = sess.run([train_op, loss_op],
                                   feed_dict={batch_images: batch_x, batch_labels: batch_y})
            train_loss, train_acc = self.evaluate_in_batch(x_train, y_train, sess, loss_op, predict_op, batch_images,
                                                           batch_labels)
            test_loss, test_acc = self.evaluate_in_batch(x_test, y_test, sess, loss_op, predict_op, batch_images,
                                                         batch_labels)
            learn_rate = sess.run(optimizer._lr)
            print('Epoch %3d: train loss %.6f, train acc %.6f, test loss %.6f, test acc %.6f, lr %.6f'
                  % (epoch, train_loss, train_acc, test_loss, test_acc, learn_rate))

    def evaluate_in_batch(self, x, y, sess, loss_op, predict_op, x_placeholder, y_placeholder):
        n_samples = len(x)
        n_iterations = int(ceil(n_samples / float(self.batch_size)))
        y_pred = np.zeros([y.shape[0]])
        losses = []
        for i in range(n_iterations):
            batch_x = x[i * self.batch_size:(i + 1) * self.batch_size]
            batch_y = y[i * self.batch_size:(i + 1) * self.batch_size]
            loss, ans = sess.run([loss_op, predict_op], feed_dict={x_placeholder: batch_x, y_placeholder: batch_y})
            losses.append(loss)
            y_pred[i * self.batch_size:(i + 1) * self.batch_size] = ans
        accuaracy = np.mean(np.equal(y_pred, np.argmax(y, axis=1)).astype(np.float32))
        return np.sum(losses) / n_samples, accuaracy


if __name__ == '__main__':
    solver = Solver()
    solver.train()
