import tensorflow as tf
from tensorflow.python.ops import array_ops


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


class HyperCell(object):
    def __init__(self, f_size, in_size, out_size, z_dim, name):
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.z_dim = z_dim
        self.name = name
        # create hypernetwork params
        kernel_initializer = tf.orthogonal_initializer(1.0)
        bias_initializer = tf.constant_initializer(0.0)
        with tf.variable_scope(self.name):
            w1 = tf.get_variable('w1', shape=[self.z_dim, self.in_size * self.z_dim],
                                 dtype=tf.float32, initializer=kernel_initializer)
            b1 = tf.get_variable('b1', shape=[self.in_size * self.z_dim],
                                 dtype=tf.float32, initializer=bias_initializer)
            w2 = tf.get_variable('w2', shape=[self.z_dim, self.f_size * self.out_size * self.f_size],
                                 dtype=tf.float32, initializer=kernel_initializer)
            b2 = tf.get_variable('b2', shape=[self.f_size * self.out_size * self.f_size],
                                 dtype=tf.float32, initializer=bias_initializer)

    def conv2d(
            self,
            inputs,
            filters,
            kernel_size,
            stride=(1, 1),
            padding='valid',
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            scope=None):
        if kernel_size != self.f_size and kernel_size != [self.f_size, self.f_size]:
            raise Exception('kernel_size must be the same with f_size')
        dim_in = inputs.get_shape().as_list()[-1]
        # create conv weight
        conv_weight = self._create_conv_weight(dim_in, filters, scope, kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer)
        outputs = tf.nn.conv2d(inputs, conv_weight, strides=stride, padding=padding, data_format='NHWC')
        if use_bias:
            bias = tf.get_variable('bias', filters)
            outputs += bias
        return outputs

    def conv2d_same(self, inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
        if stride == 1:
            return self.conv2d(
                inputs,
                num_outputs,
                kernel_size,
                stride=1,
                # rate=rate,
                padding='SAME',
                scope=scope)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = array_ops.pad(
                inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return self.conv2d(
                inputs,
                num_outputs,
                kernel_size,
                stride=stride,
                # rate=rate,
                padding='VALID',
                scope=scope
            )

    def _create_conv_weight(self, dim_in, dim_out, name, kernel_initializer=None, bias_initializer=None):
        if dim_in % self.in_size != 0:
            raise Exception('dim_in%%in_size=%d' % (dim_in % self.in_size))
        if dim_out % self.out_size != 0:
            raise Exception('dim_out%%out_size=%d' % (dim_out % self.out_size))
        in_list = []

        for i in range(dim_in // self.in_size):
            out_list = []
            for j in range(dim_out // self.out_size):
                with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                    # create embedding
                    z = tf.get_variable('%s_%d_%d' % (name, i, j), [1, self.z_dim], tf.float32)
                    # load hypyer params
                    w1 = tf.get_variable('w1', shape=[self.z_dim, self.in_size * self.z_dim],
                                         dtype=tf.float32, initializer=kernel_initializer)
                    b1 = tf.get_variable('b1', shape=[self.in_size * self.z_dim],
                                         dtype=tf.float32, initializer=bias_initializer)
                    w2 = tf.get_variable('w2', shape=[self.z_dim, self.f_size * self.out_size * self.f_size],
                                         dtype=tf.float32, initializer=kernel_initializer)
                    b2 = tf.get_variable('b2', shape=[self.f_size * self.out_size * self.f_size],
                                         dtype=tf.float32, initializer=bias_initializer)
                    # create conv weight
                    a = tf.matmul(z, w1) + b1
                    a = tf.reshape(a, [self.in_size, self.z_dim])
                    weight = tf.matmul(a, w2) + b2
                    weight = tf.reshape(weight, [self.in_size, self.out_size, self.f_size, self.f_size])
                    weight = tf.transpose(weight, [2, 3, 0, 1])  # (f_size, f_size, in_size, out_size)
                    out_list.append(weight)
                # concat
                out_weight = tf.concat(out_list, axis=3)
                in_list.append(out_weight)
        # concat
        conv_weight = tf.concat(in_list, axis=2)
        return conv_weight
