from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import tensorflow as tf
import numpy as np

from dataset import Dataset, DataIterator
from loss import binary_cross_entropy_with_logits

def downsample(I): # 1/2 I
    shape = I.get_shape() # [batch, height, width, channels]
    h = shape[1]
    w = shape[2]
    h2 = int(h // 2)
    w2 = int(w // 2)
    return tf.image.resize_images(I, h2, w2, tf.image.ResizeMethod.BILINEAR)

def upsample(I): # 2 I
    shape = I.get_shape() # [batch, height, width, channels]
    h = shape[1]
    w = shape[2]
    h2 = int(h * 2)
    w2 = int(w * 2)
    return tf.image.resize_images(I, h2, w2, tf.image.ResizeMethod.BILINEAR)

def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b

def dense_layer(x, W_shape, b_shape, activation, name=None):
    with tf.name_scope(name):
        W, b = weight_bias(W_shape, b_shape)
        return activation(tf.matmul(x, W) + b)

def conv2d_layer(x, W_shape, b_shape, strides, padding, name=None):
    with tf.name_scope(name):
        W, b = weight_bias(W_shape, b_shape)
        return tf.nn.relu(tf.nn.conv2d(x, W, strides, padding) + b)

def generator(l):
    print('G', l.get_shape())
    h0 = conv2d_layer(l, [7, 7, 4, 64], [64], [1, 2, 2, 1], 'SAME', name='h0')
    h1 = conv2d_layer(h0, [7, 7, 64, 64 * 2], [64 * 2], [1, 2, 2, 1], 'SAME', name='h1')
    h2 = conv2d_layer(h1, [5, 5, 64 * 2, 64 * 3], [64 * 3], [1, 2, 2, 1], 'SAME', name='h2')
    return tf.reshape(h2, [-1, 32, 32, 3])

def discriminator(h, keep_prob):
    print('D', h.get_shape())
    h0 = conv2d_layer(h, [5, 5, 3, 64], [64], [1, 2, 2, 1], 'SAME', name='h0')
    h1 = conv2d_layer(h0, [5, 5, 64, 64], [64], [1, 2, 2, 1], 'SAME', name='h1')
    d = tf.nn.dropout(h1, keep_prob)
    r = tf.reshape(d, [-1, 8 * 8 * 64])
    return dense_layer(r, [8 * 8 * 64, 1], [1], tf.sigmoid, name='p')

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

    train_g = tf.train.GradientDescentOptimizer(1e-1).minimize(loss_g, var_list=vars_g)
    train_d = tf.train.GradientDescentOptimizer(1e-5).minimize(loss_d, var_list=vars_d)

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
