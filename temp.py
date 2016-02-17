from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import tensorflow as tf
import numpy as np

from dataset import Dataset, DataIterator

batch_size = 128
num_epochs = 2048
checkpoint_interval = 16

image_size = (32, 32)

with tf.Session() as sess:
    h = tf.placeholder(tf.float32, (None,) + image_size + (3,), name='h')
    l = tf.placeholder(tf.float32, (None,) + image_size + (4,), name='l')

    with tf.variable_scope('generator'):
        h_ = generator(l)

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    with tf.variable_scope('discriminator'):
        p = discriminator(h, keep_prob) # target: 1

    with tf.variable_scope('discriminator', reuse=True):
        p_ = discriminator(h_, keep_prob) # target: 0

    # maximize log(D(G(z)))
    loss_g = tf.reduce_mean(-tf.log(p_ + 1e-12))

    # maximize log(D(x)) + log(1 - D(G(z)))
    loss_d_real = tf.reduce_mean(-tf.log(p + 1e-12))
    loss_d_fake = tf.reduce_mean(-tf.log((1. - p_) + 1e-12))
    loss_d = loss_d_real + loss_d_fake

    tf.image_summary("l", l * 255.0)
    tf.image_summary("h", h + 0.5)
    tf.image_summary("h_", h_ + 0.5)

    tf.histogram_summary("p", p)
    tf.histogram_summary("p_", p_)

    tf.scalar_summary("loss_d_real", loss_d_real)
    tf.scalar_summary("loss_d_fake", loss_d_fake)
    tf.scalar_summary("loss_d", loss_d)
    tf.scalar_summary("loss_g", loss_g)

    vars_g = [var for var in tf.trainable_variables() if 'generator' in var.name]
    vars_d = [var for var in tf.trainable_variables() if 'discrimin' in var.name]

    train_g = tf.train.GradientDescentOptimizer(2e-4).minimize(loss_g, var_list=vars_g)
    train_d = tf.train.GradientDescentOptimizer(2e-4).minimize(loss_d, var_list=vars_d)

    merged_summaries = tf.merge_all_summaries()

    sess.run(tf.initialize_all_variables())

    dataset = Dataset("cifar10/")
    dataset_iter = DataIterator(dataset.train_images, dataset.train_labels, batch_size)

    print('train', dataset.train_images.shape, dataset.train_labels.shape)
    print('valid', dataset.valid_images.shape, dataset.valid_labels.shape)
    print('test', dataset.test_images.shape, dataset.test_labels.shape)

    summary_writer = tf.train.SummaryWriter('logs_{0}/'.format(int(time.time())), sess.graph_def)

    for epoch in range(num_epochs):
        if epoch % checkpoint_interval == 0:
            I0 = dataset.valid_images / 255.0
            I1 = downsample(tf.constant(I0, tf.float32))
            l0 = sess.run(upsample(I1))
            h0 = I0 - l0

            z0 = np.random.uniform(-1.0, 1.0, I0.shape[:-1] + (1,)).astype(np.float32)
            # z0 = np.ones(I0.shape[:-1] + (1,), dtype=np.float32)
            l0 = np.concatenate([l0, z0], axis=-1)

            valid_loss_g, summary = sess.run([loss_g, merged_summaries], feed_dict={ l: l0, h: h0, keep_prob: 1.0 })
            summary_writer.add_summary(summary, epoch)

            valid_loss_d, summary = sess.run([loss_d, merged_summaries], feed_dict={ l: l0, h: h0, keep_prob: 1.0 })
            summary_writer.add_summary(summary, epoch)

            print("[{0}] valid loss: {1} (d) {2} (g)".format(epoch, valid_loss_d, valid_loss_g))

        batch_images, _ = dataset_iter.next_batch()

        I0 = batch_images / 255.0
        I1 = downsample(tf.constant(I0, tf.float32))
        l0 = sess.run(upsample(I1))
        h0 = I0 - l0

        # noise input zk to Gk is presented as a 4th "color plane" to low-pass lk
        z0 = np.random.uniform(-1.0, 1.0, (batch_size,) + image_size + (1,)).astype(np.float32)
        # z0 = np.ones((batch_size,) + image_size + (1,), dtype=np.float32)
        l0 = np.concatenate([l0, z0], axis=-1)

        sess.run(train_g, feed_dict={ l: l0, h: h0, keep_prob: 0.5 })
        sess.run(train_d, feed_dict={ l: l0, h: h0, keep_prob: 0.5 })

        print('[{0}] training...'.format(epoch))
