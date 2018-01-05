import numpy as np
import tensorflow as tf


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def orthogonal_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape) * scale, dtype)

    return _initializer


def super_linear(x, output_size, scope=None, reuse=False, init_w="ortho", weight_start=0.0, use_bias=True,
                 bias_start=0.0):
    # support function doing linear operation.  uses ortho initializer defined earlier.
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or "linear"):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()

        w_init = None  # uniform
        x_size = shape[1]
        h_size = output_size
        if init_w == "zeros":
            w_init = tf.constant_initializer(0.0)
        elif init_w == "constant":
            w_init = tf.constant_initializer(weight_start)
        elif init_w == "gaussian":
            w_init = tf.random_normal_initializer(stddev=weight_start)
        elif init_w == "ortho":
            w_init = orthogonal_initializer(1.0)

        w = tf.get_variable("super_linear_w",
                            [shape[1], output_size], tf.float32, initializer=w_init)
        if use_bias:
            b = tf.get_variable("super_linear_b", [output_size], tf.float32,
                                initializer=tf.constant_initializer(bias_start))
            return tf.matmul(x, w) + b
        return tf.matmul(x, w)
