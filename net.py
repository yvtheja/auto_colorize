import tensorflow as tf

class Net:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_kernel_size(self, upscale_factor):
        return 2 * upscale_factor - upscale_factor % 2

    def conv_layer(self, X, filter_shape, strides, name, is_training=True, act=tf.nn.relu, padding='SAME',
                   max_pool=False, pool_factor=2):
        with tf.variable_scope(name):
            filter = tf.get_variable('weights', initializer=tf.truncated_normal(filter_shape, stddev=0.2))

            maps_ = tf.nn.conv2d(X, filter, strides, padding=padding)

            maps = self.batch_norm(maps_, is_training=is_training)

            maps = act(maps) if act is not None else maps

            maps = tf.nn.max_pool(maps, [1, 3, 3, 1], [1, pool_factor, pool_factor, 1], padding='SAME') \
                if max_pool is True else maps

        return maps

    def deconv_layer(self, X, upscale_factor, in_maps, out_maps, name, is_training, act=tf.nn.relu):
        with tf.variable_scope(name):
            input_shape = X.get_shape().as_list()

            filter_size = self.get_kernel_size(upscale_factor)
            filter_shape = [filter_size, filter_size, out_maps, in_maps]

            new_width = input_shape[1] * upscale_factor
            new_height = input_shape[2] * upscale_factor

            filter = tf.get_variable('weights', initializer=tf.truncated_normal(filter_shape, stddev=0.2))

            maps_ = tf.nn.conv2d_transpose(X, filter,
                                          output_shape=[self.batch_size, new_height, new_width, out_maps],
                                          strides=[1, upscale_factor, upscale_factor, 1])

            maps = self.batch_norm(maps_, is_training=is_training)

            maps = act(maps) if act is not None else maps

        return maps

    def fully_conn_layer(self, X_flat, n_rows, nl, name, is_training=True, act=tf.nn.relu):
        with tf.variable_scope(name):
            filter = tf.get_variable('fc_weights', initializer=tf.truncated_normal([n_rows, nl], stddev=0.2))

            z_ = tf.matmul(X_flat, filter)

            z = self.batch_norm(z_, fully_conn=True, is_training=is_training)

            a = act(z) if act is not None else z

        return a


    def batch_norm(self, maps, is_training, decay=0.999, epsilon=0.001, fully_conn=False):
        scale = tf.get_variable('gamm_bn', initializer=tf.ones([maps.get_shape()[-1]]))
        offset = tf.get_variable('offset_bn', initializer=tf.zeros([maps.get_shape()[-1]]))

        pop_mean = tf.get_variable('mean_ewa', initializer=tf.zeros([maps.get_shape()[-1]]), trainable=False)
        pop_variance = tf.get_variable('variance_ewa', initializer=tf.ones([maps.get_shape()[-1]]), trainable=False)

        if is_training:
            if fully_conn is True:
                avg_axis = [0]
            else:
                avg_axis = [0, 1, 2]

            batch_mean, batch_variance = tf.nn.moments(maps, avg_axis)

            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(maps, batch_mean, batch_variance, offset, scale, epsilon)

        else:
            return tf.nn.batch_normalization(maps, pop_mean, pop_variance, offset, scale, epsilon)

