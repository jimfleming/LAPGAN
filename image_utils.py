import scipy.misc
import numpy as np

import tensorflow as tf

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

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return scipy.misc.imsave(path, img)

def inverse_transform(images):
    return (images + 1.) / 2.
