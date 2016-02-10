from __future__ import division

from sklearn.cross_validation import train_test_split
import tensorflow as tf
import numpy as np
import cPickle

def unpickle(path):
    with open(path, 'rb') as f:
        return cPickle.load(f)

def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class DataIterator:

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_examples = num_examples = x.shape[0]
        self.num_batches = num_examples // batch_size
        self.pointer = 0
        assert self.batch_size <= self.num_examples

    def next_batch(self):
        start = self.pointer
        self.pointer += self.batch_size

        if self.pointer > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)

            self.x = self.x[perm]
            self.y = self.y[perm]

            start = 0
            self.pointer = self.batch_size

        end = self.pointer
        return self.x[start:end], self.y[start:end]

    def iterate(self):
        for step in range(self.num_batches):
            yield self.next_batch()

class Dataset:

    def __init__(self, dataset_path):
        train_batch_1 = unpickle("{0}/data_batch_1".format(dataset_path))
        train_batch_2 = unpickle("{0}/data_batch_2".format(dataset_path))
        train_batch_3 = unpickle("{0}/data_batch_3".format(dataset_path))
        train_batch_4 = unpickle("{0}/data_batch_4".format(dataset_path))
        train_batch_5 = unpickle("{0}/data_batch_5".format(dataset_path))

        train_data = np.concatenate([
            train_batch_1['data'],
            train_batch_2['data'],
            train_batch_3['data'],
            train_batch_4['data'],
            train_batch_5['data']
        ], axis=0)
        train_labels = np.concatenate([
            train_batch_1['labels'],
            train_batch_2['labels'],
            train_batch_3['labels'],
            train_batch_4['labels'],
            train_batch_5['labels']
        ], axis=0)

        train_images = np.swapaxes(train_data.reshape([-1, 32, 32, 3], order='F'), 1, 2)

        test_batch = unpickle("{0}/test_batch".format(dataset_path))
        test_data = test_batch['data']
        test_images = np.swapaxes(test_data.reshape([-1, 32, 32, 3], order='F'), 1, 2)
        test_labels = np.array(test_batch['labels'])

        train_images, valid_images, train_labels, valid_labels = \
            train_test_split(train_images, train_labels, test_size=0.01, random_state=42)

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images

        self.train_labels = one_hot(train_labels, 10)
        self.valid_labels = one_hot(valid_labels, 10)
        self.test_labels = one_hot(test_labels, 10)
