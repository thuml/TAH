import os
import sys
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import pdb
from datetime import datetime
from math import ceil
import random
import cv2


class Alexnet(object):
    def __init__(self, model, hash_bit_num, sess):
        self._initialized = False
        self._lr_mult = None
        self._model_path = model
        self._output_dim = hash_bit_num
        self._sess = sess

    def extract(self, img, train_phase=False):
        '''
        alexnet structure
        Args:
            img: [batch_size, w, h, c] 4-D tensor
        Return:
            hash bits: [batch_size, output_dim] tensor
        '''
        lr_mult = dict()

        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        
        ### Conv1
        ### Output 96, kernel 11, stride 4
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', [11, 11, 3, 96], initializer=tf.constant_initializer(0))
            biases = tf.get_variable('biases', [96], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            conv = tf.nn.conv2d(img, kernel, [1, 4, 4, 1], padding='VALID')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        
        ### Pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                                    ksize=[1,3,3,1],
                                    strides=[1,2,2,1],
                                    padding='VALID',
                                    name='pool1')
        ### LRN1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)
        ### Conv2
        ### Output 256, pad 2, kernel 5, group 2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', [5,5,48,256], initializer=tf.constant_initializer(0))
            biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.lrn1, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        
        ### Pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool2')
        ### LRN2
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)
        ### Conv3
        ### Output 384, pad 1, kernel 3
        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', [3,3,256,384], initializer=tf.constant_initializer(0))
            biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Conv4
        ### Output 384, pad 1, kernel 3, group 2
        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', [3,3,192,384], initializer=tf.constant_initializer(0))
            biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.conv3, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Conv5
        ### Output 256, pad 1, kernel 3, group 2
        with tf.variable_scope('conv5') as scope:
            kernel = tf.get_variable('weights', [3,3,192,256], initializer=tf.constant_initializer(0))
            biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.conv4, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out)
            lr_mult[kernel] = 1
            lr_mult[biases] = 2
        ### Pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')
        ### FC6
        ### Output 4096
        with tf.variable_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.get_variable('weights', [9216,4096], initializer=tf.constant_initializer(0))
            fc6b = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            if train_phase:
                self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
            else:
                self.fc6 = tf.nn.relu(fc6l)
            lr_mult[fc6w] = 1
            lr_mult[fc6b] = 2
            
        ### FC7
        ### Output 4096
        with tf.variable_scope('fc7') as scope:
            fc7w = tf.get_variable('weights', [4096,4096], initializer=tf.constant_initializer(0))
            fc7b = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            if train_phase:
                self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
            else:
                self.fc7 = tf.nn.relu(fc7l)
            lr_mult[fc7w] = 1
            lr_mult[fc7b] = 2
        
        ### Hash Layer
        ### Output out_dim
        with tf.variable_scope("fc8") as scope:
            fc8w = tf.get_variable("weights", [4096,48], initializer=tf.constant_initializer(0))
            fc8b = tf.get_variable("biases", [self._output_dim], initializer=tf.constant_initializer(0))
            saver = tf.train.Saver()
            saver.restore(self._sess, self._model_path)
            self.fc8 = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            lr_mult[fc8w] = 10
            lr_mult[fc8b] = 20
        
        self._initialized == True
        self._lr_mult = lr_mult
        return self.fc8

    @property
    def lr_mult(self):
        assert self._initialized == True, "Alexnet not initialized"
        return self._lr_mult

    @property
    def var_list(self):
        return self._lr_mult.keys()

    @property
    def output_dim(self):
        return self._output_dim
