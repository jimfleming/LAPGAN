from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import tensorflow as tf
import numpy as np

from dataset import Dataset, DataIterator
from model import LAPGAN
from image_utils import save_images

def main():
    with tf.Session() as sess:
        num_epoch = 5
        checkpoint_interval = 10

        batch_size = 64
        image_size = 32

        model = LAPGAN(sess, batch_size=batch_size)

        dataset = Dataset("cifar10/")
        dataset_iter = DataIterator(dataset.train_images, dataset.train_labels, batch_size)

        summary_writer = tf.train.SummaryWriter('logs_{0}/'.format(int(time.time())), sess.graph_def)

        sess.run(tf.initialize_all_variables())

        sample_images = dataset.valid_images[:model.sample_size].astype(np.float32) / 255.0
        sample_z = np.random.uniform(-1.0, 1.0, size=(model.sample_size , model.z_dim))

        d_overpowered = False

        step = 0
        for epoch in range(num_epoch):
            for batch_images, _ in dataset_iter.iterate():
                # I0 = batch_images / 255.0
                # I1 = downsample(tf.constant(I0, tf.float32))
                # l0 = sess.run(upsample(I1))
                # h0 = I0 - l0
                # z0 = np.random.uniform(-1.0, 1.0, (batch_size,) + image_size + (1,)).astype(np.float32)
                # l0 = np.concatenate([l0, z0], axis=-1)

                batch_images = batch_images.astype(np.float32) / 255.0
                batch_z = np.random.uniform(-1.0, 1.0, [batch_size, model.z_dim]).astype(np.float32)

                # update d network
                if not d_overpowered:
                    sess.run(model.d_optim, feed_dict={ model.x: batch_images, model.z: batch_z })

                # update g network
                sess.run(model.g_optim, feed_dict={ model.z: batch_z })

                if step % checkpoint_interval == 0:
                    # I0 = dataset.valid_images / 255.0
                    # I1 = downsample(tf.constant(I0, tf.float32))
                    # l0 = sess.run(upsample(I1))
                    # h0 = I0 - l0
                    # z0 = np.random.uniform(-1.0, 1.0, I0.shape[:-1] + (1,)).astype(np.float32)
                    # l0 = np.concatenate([l0, z0], axis=-1)

                    batch_images = dataset.valid_images[:batch_size].astype(np.float32) / 255.0
                    batch_z = np.random.uniform(-1.0, 1.0, [batch_size, model.z_dim]).astype(np.float32)

                    d_loss, g_loss, summary = sess.run([
                        model.d_loss,
                        model.g_loss,
                        model.merged
                    ], feed_dict={
                        model.x: batch_images,
                        model.z: batch_z
                    })

                    d_overpowered = d_loss < g_loss / 2

                    samples = sess.run(model.G, feed_dict={
                        model.x: sample_images,
                        model.z: sample_z
                    })

                    summary_writer.add_summary(summary, step)
                    save_images(samples, [8, 8], './samples/train_{0}_{1}.png'.format(epoch, step))
                    print('[{0}, {1}] loss: {2} (D) {3} (G) (d overpowered?: {4})'.format(epoch, step, d_loss, g_loss, d_overpowered))

                step += 1

if __name__ == '__main__':
    main()
