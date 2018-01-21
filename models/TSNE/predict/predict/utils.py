import os
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import time
import pdb
from datetime import datetime
from math import ceil
import random
import cv2

def log(s1, s2):
    print ("%s #%s# %s" % (datetime.now(), s1, s2))

### Define custom py_func which takes also a grad op as argument:
### Source code: https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    ### Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

class Dataset(object):
    def __init__(self, path, n_class, modal='image', name='None'):
        with open(path, 'r') as fin:
            self._n_class = n_class
            self._one_hot_label = False
            self._lines = fin.readlines()
            self._n_samples = len(self._lines)
            self._name = name
            self._samples = [None] * self._n_samples
            self._labels = [None] * self._n_samples
            self._perm = np.arange(self._n_samples)
            #np.random.shuffle(self._perm)
            self._index = 0
            self._loaded = 0

    def single_data(self, index):
        image = cv2.resize(cv2.imread(self._lines[index].strip().split()[0]), (256, 256))
        if self._one_hot_label:
            label = int(self._lines[index].strip().split()[1])
            label = [int(label == i) for i in xrange(self._n_class)]
        else:
            label = [int(i) for i in self._lines[index].strip().split()[1:]]
        return image, label

    def get_data(self, indices):
        if self._loaded != self._n_samples:
            data = []
            label = []
            for i in indices:
                if self._samples[i] is None:
                    self._samples[i], self._labels[i] = self.single_data(i)
                    self._loaded += 1
                data.append(self._samples[i])
                label.append(self._labels[i])
            if self._loaded == self._n_samples:
                self._samples = np.asarray(self._samples)
                self._labels = np.asarray(self._labels)
            return np.asarray(data), np.asarray(label)
        else:
            return self._samples[indices, :], self._labels[indices, :]

    def full_data(self):
        return self.get_data(np.arange(self._n_samples))

    def next_batch(self, batch_size):
        start = self._index
        self._index += batch_size
        if self._index >= self._n_samples:
            has_next = False
            end = self._n_samples
        else:
            has_next = True
            end = self._index
        img, label = self.get_data(self._perm[start:end])
        if has_next == False:
            ones = [1] * (end - start)
            rep_num = batch_size - (end - start) + 1
            ones[-1] = rep_num
            img = np.repeat(img, ones, axis = 0)
            label = np.repeat(label, ones, axis = 0)
        return img, label, has_next
