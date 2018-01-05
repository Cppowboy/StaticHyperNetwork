# includes
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datasets.dataset import read_data_sets
from utils.visualize import show_filter
from model.mnist import MNIST

# misc
np.set_printoptions(precision=5, edgeitems=8, linewidth=200)


# training
def process_epoch(sess, model, dataset, train_mode=False, print_every=0):
    num_examples = dataset.num_examples
    batch_size = hps_model.batch_size
    total_batch = int(num_examples / batch_size)

    avg_loss = 0.
    avg_pred_error = 0.
    lr = model.hps.lr

    for i in range(total_batch):
        batch_images, batch_labels = dataset.next_batch(batch_size, with_label=True, one_hot=False)

        if train_mode:
            loss, pred, lr = model.partial_train(sess, batch_images, np.eye(dataset.num_classes)[batch_labels])
            model.update_lr(sess)
        else:
            loss, pred = model.partial_eval(sess, batch_images, np.eye(dataset.num_classes)[batch_labels])

        pred_error = 1.0 - np.sum((pred == batch_labels)) / float(batch_size)

        if print_every > 0 and i > 0 and i % print_every == 0:
            print "Batch:", '%d' % (i), \
                "/", '%d' % (total_batch), \
                "loss=", "{:.4f}".format(loss), \
                "err=", "{:.4f}".format(pred_error)

        assert (loss < 1000000)  # make sure it is not NaN or Inf

        avg_loss += loss / num_examples * batch_size
        avg_pred_error += pred_error / num_examples * batch_size
    return avg_loss, avg_pred_error, lr


def train_model(sess, model, eval_model, mnist, num_epochs, save_model=True):
    # train the model for num_epochs

    best_valid_loss = 100.
    best_valid_pred_error = 1.0
    eval_loss = 100.
    eval_pred_error = 1.0

    for epoch in range(num_epochs):

        train_loss, train_pred_error, lr = process_epoch(sess, model, mnist.train, train_mode=True, print_every=10)

        valid_loss, valid_pred_error, _ = process_epoch(sess, eval_model, mnist.valid, train_mode=False)

        if valid_pred_error <= best_valid_pred_error:
            best_valid_pred_error = valid_pred_error
            best_valid_loss = valid_loss
            eval_loss, eval_pred_error, _ = process_epoch(sess, eval_model, mnist.test, train_mode=False)

            if (save_model):
                model.save_model(sess, epoch)

        print "Epoch:", '%d' % (epoch), \
            "train_loss=", "{:.4f}".format(train_loss), \
            "train_err=", "{:.4f}".format(train_pred_error), \
            "valid_err=", "{:.4f}".format(valid_pred_error), \
            "best_valid_err=", "{:.4f}".format(best_valid_pred_error), \
            "test_err=", "{:.4f}".format(eval_pred_error), \
            "lr=", "{:.6f}".format(lr)


class HParams(object):
    pass


hps_model = HParams()
hps_model.lr = 0.005
hps_model.lr_decay = 0.999
hps_model.min_lr = 0.0001
hps_model.is_training = True
hps_model.x_dim = 28
hps_model.num_classes = 10
hps_model.c_dim = 1
hps_model.batch_size = 1000
hps_model.grad_clip = 100.0
hps_model.hyper_mode = False
hps_model.model_path = '/tmp/'
mnist_data = input_data.read_data_sets("/data/mnist", one_hot=False)
mnist = read_data_sets(mnist_data)
tf.reset_default_graph()
model = MNIST(hps_model)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
train_model(sess, model, model, mnist, 50, save_model=False)
t_vars = tf.trainable_variables()
count_t_vars = 0
for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    print var.name, var.get_shape(), num_param
print "total trainable variables = %d" % (count_t_vars)
conv_filter = sess.run((model.conv2_weights))
show_filter(conv_filter)
hps_model.hyper_mode = True
sess.close()
tf.reset_default_graph()
model = MNIST(hps_model)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
train_model(sess, model, model, mnist, 50, save_model=False)
conv_filter = sess.run((model.conv2_weights))
t_vars = tf.trainable_variables()
count_t_vars = 0
for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    print var.name, var.get_shape(), num_param
print "total trainable variables = %d" % (count_t_vars)
show_filter(conv_filter)
sess.close()
