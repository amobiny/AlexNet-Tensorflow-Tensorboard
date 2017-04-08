import numpy as np
from six.moves import cPickle as pickle
from tensorflow.examples.tutorials.mnist import input_data


def load_train_data(image_size, num_channels, label_cnt):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_dataset, train_labels, valid_dataset, valid_labels = mnist.train.images, mnist.train.labels,\
    mnist.validation.images, mnist.validation.labels
    train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_channels, label_cnt)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_channels, label_cnt)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels


def load_test_data(image_size, num_channels, label_cnt):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_dataset, test_labels = mnist.test.images, mnist.test.labels
    test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_channels, label_cnt)
    print('Test set', test_dataset.shape, test_labels.shape)
    return test_dataset, test_labels


def reformat(dataset, labels, image_size, num_channels, label_cnt):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
#    labels = (np.arange(label_cnt) == labels[:, None]).astype(np.float32)
    labels = labels
    return dataset, labels
