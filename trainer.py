import tensorflow as tf
from utility.image_conversions import ImageConversions
import glob
import generator
import discriminator
import numpy as np

# Global variables
batch_size = 4
max_iterations = 100000
image_height = 224
image_width = 224

gen_input_ph = tf.placeholder(tf.float32, shape=(None, image_height, image_width, 3))
dis_input_ph = tf.placeholder(tf.float32, shape=(None, image_height, image_width, 3))

def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, image_file = reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file, channels=3)
    image_resized = tf.image.resize_images(image, [image_height, image_width],
                                             method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

    return image_resized

def input_pipeline(filenames, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

    image_yuv_resized = read_my_file_format(filename_queue)
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    image_batch = tf.train.shuffle_batch([image_yuv_resized], batch_size=batch_size, capacity=capacity,
                                         min_after_dequeue=min_after_dequeue)
    #image_batch_yuv = ImageConversions.rgb_to_yuv(image_batch)

    return image_batch

with tf.variable_scope('input_pipeline'):
    image_filenames = glob.glob('../training_data/n01440764/*.jpeg')
    image_batch_rgb = input_pipeline(image_filenames) / 225
    image_gray_batch = tf.slice(image_batch_rgb, [0, 0, 0, 0], [batch_size, image_height, image_width, 1])
    image_gray_batch_ip = tf.concat([image_gray_batch, image_gray_batch, image_gray_batch], axis=3)

    image_batch_rgb_test = input_pipeline(image_filenames) / 225
    image_gray_batch_test = tf.slice(image_batch_rgb_test, [0, 0, 0, 0], [batch_size, image_height, image_width, 1])
    image_gray_batch_ip_test = tf.concat([image_gray_batch_test, image_gray_batch_test, image_gray_batch_test], axis=3)

with tf.variable_scope('generator'):
    gen = generator.Generator(batch_size, image_height, image_width)
    gen_op = gen.generate_sample(gen_input_ph)
    gen_op_test = gen.generate_sample(gen_input_ph, reuse=True, is_training=False)
    tf.summary.image('generated_samples', gen_op_test, max_outputs=4)

with tf.variable_scope('discriminator'):
    dis = discriminator.Discriminator(batch_size)
    dis_fake = dis.discriminate(gen_op)
    dis_fake_test = dis.discriminate(gen_op_test, reuse=True, is_training=False)
    dis_real = dis.discriminate(dis_input_ph, reuse=True)
    dis_real_test = dis.discriminate(dis_input_ph, reuse=True, is_training=False)

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake,
                                                                  labels=tf.ones_like(dis_fake),
                                                                  name='gen_loss'))
dis_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake,
                                                                       labels=tf.zeros_like(dis_fake),
                                                                       name='dis_fake_loss'))
dis_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real,
                                                                       labels=tf.fill([batch_size, 1], 0.9),
                                                                       name='dis_real_loss'))
dis_loss = dis_fake_loss + dis_real_loss

gen_loss_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_test,
                                                                  labels=tf.ones_like(dis_fake_test),
                                                                  name='gen_loss'))
tf.summary.scalar('gen_loss', gen_loss_test)
dis_fake_loss_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_test,
                                                                       labels=tf.zeros_like(dis_fake_test),
                                                                       name='dis_fake_loss'))
tf.summary.scalar('dis_fake_loss', dis_fake_loss_test)
dis_real_loss_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_test,
                                                                       labels=tf.ones_like(dis_real_test),
                                                                       name='dis_real_loss'))
tf.summary.scalar('dis_real_loss', dis_real_loss_test)
dis_loss_test = dis_fake_loss_test + dis_real_loss_test

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'discriminator' in var.name]
g_vars = [var for var in tvars if 'generator' in var.name]

with tf.variable_scope('alpha_decay'):
    global_step_dis_fake = tf.Variable(0, trainable=False)
    global_step_dis_real = tf.Variable(0, trainable=False)
    global_step_gen = tf.Variable(0, trainable=False)

    learning_rate_dis_fake = tf.train.exponential_decay(0.05, global_step_dis_fake, decay_steps=1000,
                                                        decay_rate=0.96, staircase=True)
    learning_rate_dis_real = tf.train.exponential_decay(0.05, global_step_dis_real, decay_steps=1000,
                                                        decay_rate=0.96, staircase=True)
    learning_rate_gen = tf.train.exponential_decay(0.05, global_step_gen, decay_steps=1000,
                                                        decay_rate=0.96, staircase=True)

with tf.variable_scope('adam_optimizer'):
    dis_fake_trainer = tf.train.AdamOptimizer(learning_rate_dis_fake).minimize(dis_fake_loss,
                                                                               var_list=d_vars,
                                                                               global_step=global_step_dis_fake)
    dis_real_trainer = tf.train.AdamOptimizer(learning_rate_dis_real).minimize(dis_real_loss,
                                                                               var_list=d_vars,
                                                                               global_step=global_step_dis_real)
    gen_trainer = tf.train.AdamOptimizer(learning_rate_gen).minimize(gen_loss,
                                                                   var_list=g_vars,
                                                                   global_step=global_step_gen)

saver = tf.train.Saver()

with tf.Session() as sess:
    gen_loss_np = 0
    dis_fake_loss_np, dis_real_loss_np = 1, 1
    dis_real_update_count, d_fake_update_count, g_update_count = 0, 0, 0
    tf.summary.scalar('dis_real_update_count', dis_real_update_count)
    tf.summary.scalar('d_fake_update_count', d_fake_update_count)
    tf.summary.scalar('g_update_count', g_update_count)

    init_op = tf.global_variables_initializer(), tf.local_variables_initializer()
    summaries = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter('./log/training_data', sess.graph)
    writer_test = tf.summary.FileWriter('./log/test_data', sess.graph)
    # writer_gen = tf.summary.FileWriter('./log/gen', sess.graph)
    # writer_dis_fake = tf.summary.FileWriter('./log/dis_fake', sess.graph)
    # writer_dis_real = tf.summary.FileWriter('./log/dis_real', sess.graph)
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(max_iterations):
        image_batch_np = image_gray_batch_ip.eval()
        image_batch_rgb_np = image_batch_rgb.eval()
        # print(sess.run(image_batch).shape)
        if dis_fake_loss_np > 0.6:
            # Train discriminator on generated images
            _, dis_fake_loss_np = sess.run([dis_fake_trainer, dis_fake_loss],
                                           {gen_input_ph: image_batch_np})
            d_fake_count += 1

        if gLoss > 0.5:
            # Train the generator
            _, gLoss = sess.run([gen_trainer, gen_loss],
                                {gen_input_ph: image_batch_np})
            g_count += 1

        if dLossReal > 0.45:
            # If the discriminator classifies real images as fake,
            # train discriminator on real values
            _, dLossReal= sess.run([dis_real_trainer, dis_real_loss],
                                   {dis_input_ph: image_batch_rgb_np})
            d_real_count += 1

        if i % 2 == 0:
            summary_train = sess.run(summaries, {gen_input_ph: image_batch_np, dis_input_ph: image_batch_rgb_np})
            writer_train.add_summary(summary_train, i)

            image_gray_batch_ip_test_np = image_gray_batch_ip_test.eval()
            image_batch_rgb_test_np = image_batch_rgb_test.eval()
            summary_test = sess.run(summaries, {gen_input_ph: image_gray_batch_ip_test_np, dis_input_ph: image_batch_rgb_test_np})
            writer_train.add_summary(summary_test, i)
            d_real_count, d_fake_count, g_count = 0, 0, 0

        if i % 5000 == 0:
            save_path = saver.save(sess, "./models/pretrained_gan.ckpt", global_step=i)
            print("saved to %s" % save_path)

        print(sess.run(dis_real, feed_dict={gen_input_ph: image_batch_np, dis_input_ph: image_batch_rgb_np}).shape)

