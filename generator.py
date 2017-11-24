import tensorflow as tf
import net
import vgg16
import utils

class Generator(net.Net):
    def __init__(self, batch_size, image_height, image_width):
        net.Net.__init__(self, batch_size)
        self.image_height = image_height
        self.image_width = image_width

    def generate_sample(self, X, reuse=False, is_training=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('generator'):
            with tf.variable_scope('vgg16_model'):
                vgg = vgg16.Vgg16()
                vgg.build(X)

            with tf.variable_scope('encoder'):
                self.deconv_0 = self.deconv_layer(vgg.pool5, upscale_factor=2, in_maps=512,
                                                  out_maps=512, name='deconv_0', is_training=is_training)
                self.deconv_1 = self.deconv_layer(tf.concat([self.deconv_0, vgg.pool4], axis=3), upscale_factor=2,
                                                  in_maps=1024, out_maps=256, name='deconv_1', is_training=is_training)
                self.deconv_2 = self.deconv_layer(tf.concat([self.deconv_1, vgg.pool3], axis=3), upscale_factor=2,
                                                  in_maps=512, out_maps=128, name='deconv_2', is_training=is_training)
                self.deconv_3 = self.deconv_layer(tf.concat([self.deconv_2, vgg.pool2], axis=3), upscale_factor=2,
                                                  in_maps=256, out_maps=64, name='deconv_3', is_training=is_training)
                self.deconv_4 = self.deconv_layer(tf.concat([self.deconv_3, vgg.pool1], axis=3), upscale_factor=2,
                                                  in_maps=128, out_maps=32, name='deconv_4', is_training=is_training)
                self.conv_0 = self.conv_layer(self.deconv_4, filter_shape=[3, 3, 32, 16], strides=[1, 1, 1, 1],
                                              name='conv_0', is_training=is_training)
                self.conv_1 = self.conv_layer(self.conv_0, filter_shape=[3, 3, 16, 8], strides=[1, 1, 1, 1],
                                              name='conv_1', is_training=is_training)
                self.conv_2 = self.conv_layer(self.conv_1, filter_shape=[3, 3, 8, 4], strides=[1, 1, 1, 1],
                                              name='conv_2', is_training=is_training)
                self.conv_3 = self.conv_layer(self.conv_2, filter_shape=[3, 3, 4, 2], strides=[1, 1, 1, 1],
                                              name='conv_3', is_training=is_training)
                maps_y = tf.concat([tf.slice(X, [0, 0, 0, 0],
                                             [self.batch_size, self.image_height, self.image_width, 1]), self.conv_3], axis=3)
                self.conv_4 = self.conv_layer(maps_y, filter_shape=[3, 3, 3, 3], strides=[1, 1, 1, 1],
                                              name='conv_4', is_training=is_training, act=tf.nn.sigmoid)

        return self.conv_4

