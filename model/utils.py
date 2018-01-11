import tensorflow as tf


class ConvWeight(object):
    def __init__(self, z_dim=4, in_size=16, out_size=16, f_size=7, name_scope='ConvWeight', kernel_initializer=None,
                 bias_initializer=None):
        self.z_dim = z_dim
        self.in_size = in_size
        self.out_size = out_size
        self.f_size = f_size
        self.name_scope = name_scope
        if kernel_initializer is None:
            self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)
        else:
            self.kernel_initializer = kernel_initializer
        if bias_initializer is None:
            self.bias_initializer = tf.constant_initializer(0.0)
        else:
            self.bias_initializer = bias_initializer
        # init W_i, B_i, W_out, B_out
        self.params = self._init_params()

    def _init_params(self):
        params = {}
        with tf.variable_scope(self.name_scope):
            params['w1'] = tf.get_variable('w1', shape=[self.z_dim, self.in_size * self.z_dim],
                                           dtype=tf.float32, initializer=self.kernel_initializer)
            params['b1'] = tf.get_variable('b1', shape=[self.in_size * self.z_dim],
                                           dtype=tf.float32, initializer=self.bias_initializer)
            params['w2'] = tf.get_variable('w2', shape=[self.z_dim, self.f_size * self.out_size * self.f_size],
                                           dtype=tf.float32, initializer=self.kernel_initializer)
            params['b2'] = tf.get_variable('b2', shape=[self.f_size * self.out_size * self.f_size],
                                           dtype=tf.float32, initializer=self.bias_initializer)
        return params

    def _create_conv_weight(self, z):
        a = tf.matmul(z, self.params['w1']) + self.params['b1']
        a = tf.reshape(a, [self.in_size, self.z_dim])
        weight = tf.matmul(a, self.params['w2']) + self.params['b2']
        weight = tf.reshape(weight, [self.in_size, self.out_size, self.f_size, self.f_size])
        weight = tf.transpose(weight, [2, 3, 0, 1])  # (f_size, f_size, in_size, out_size)
        return weight

    def create_conv_weight(self, dim_in, dim_out, embed_name='z_signal'):
        if dim_in % self.in_size != 0:
            raise Exception('dim_in%in_size=%d' % (dim_in % self.in_size))
        if dim_out % self.out_size != 0:
            raise Exception('dim_out%out_size=%d' % (dim_out % self.out_size))
        in_list = []
        for i in range(dim_in // self.in_size):
            out_list = []
            for j in range(dim_out // self.out_size):
                z = tf.get_variable('%s_%d_%d' % (embed_name, i, j), [1, self.z_dim], tf.float32,
                                    initializer=self.kernel_initializer)
                weight = self._create_conv_weight(z)
                out_list.append(weight)
            out_weight = tf.concat(out_list, axis=3)
            in_list.append(out_weight)
        conv_weight = tf.concat(in_list, axis=2)
        return conv_weight
