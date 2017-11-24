import tensorflow as tf
import net

class Discriminator(net.Net):
    def __init__(self, batch_size):
        net.Net.__init__(self, batch_size)

    def discriminate(self, X, reuse=False, is_training=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('discriminator'):
            self.conv_0 = self.conv_layer(X, filter_shape=[3, 3, 3, 16], strides=[1, 1, 1, 1], name='conv_0')
            self.conv_1 = self.conv_layer(self.conv_0, filter_shape=[3, 3, 16, 32], strides=[1, 1, 1, 1], name='conv_1',
                                       max_pool=True, pool_factor=2)
            self.conv_2 = self.conv_layer(self.conv_1, filter_shape=[3, 3, 32, 64], strides=[1, 1, 1, 1], name='conv_2',
                                       max_pool=True, pool_factor=2)
            self.conv_3 = self.conv_layer(self.conv_2, filter_shape=[3, 3, 64, 128], strides=[1, 1, 1, 1], name='conv_3',
                                       max_pool=True, pool_factor=2)
            self.conv_4 = self.conv_layer(self.conv_3, filter_shape=[3, 3, 128, 256], strides=[1, 1, 1, 1], name='conv_4',
                                       max_pool=True, pool_factor=2)
            self.conv_5 = self.conv_layer(self.conv_4, filter_shape=[3, 3, 256, 512], strides=[1, 1, 1, 1], name='conv_5',
                                       max_pool=True, pool_factor=2)
            n_rows = 7*7*512

            self.full_conn_0 = self.fully_conn_layer(tf.reshape(self.conv_5, [-1, n_rows]), n_rows=n_rows, nl=1024,
                                                    name='fully_conn_0')
            self.full_conn_1 = self.fully_conn_layer(self.full_conn_0, n_rows=1024, nl=1, name='fully_conn_1')

        return self.full_conn_1



