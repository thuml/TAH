##################################################################################
# 2017 01.16 Created by Shichen Liu                                              #
# Residual Transfer Network implemented by tensorflow                            #
#                                                                                #
#                                                                                #
##################################################################################

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import scipy.io as sio
import time
import pdb
from datetime import datetime
from math import ceil
import random
import cv2
from alexnet import Alexnet
from utils import *
import datetime
import os.path as opath

IMAGE_SIZE = 227

def save_code_and_label(params):
    database_code = np.sign(params['database_code'])
    validation_code = np.sign(params['validation_code'])
    database_labels = params['database_labels']
    validation_labels = params['validation_labels']
    path = params['path']
    np.save(opath.join(path, params["prefix"]+"database_code.npy"), database_code)
    np.save(opath.join(path, params["prefix"]+"database_label.npy"), database_labels)
    np.save(opath.join(path, params["prefix"]+"validation_code.npy"), validation_code)
    np.save(opath.join(path, params["prefix"]+"validation_label.npy"), validation_labels)

def load_code_and_label(params):
    path = params['path']
    params['database_code'] = np.load(opath.join(path, params["prefix"]+"database_code.npy"))
    params['database_labels'] = np.load(opath.join(path, params["prefix"]+"database_label.npy"))
    params['validation_code'] = np.load(opath.join(path, params["prefix"]+"validation_code.npy"))
    params['validation_labels'] = np.load(opath.join(path, params["prefix"]+"validation_label.npy"))



def mean_average_precision(params):
    database_code = params['database_code']
    validation_code = params['validation_code']
    database_labels = params['database_labels']
    validation_labels = params['validation_labels']
    R = params['R']
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    
    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
    
    return np.mean(np.array(APx))

class Net(object):
    def __init__(self, config):
        ### Initialize setting
        log('setup', 'Initializing network')
        np.set_printoptions(precision=4)
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.n_class = config['n_class']
        self.hash_bit_num = config['hash_bit_num']
        self.model_path = config['model_path']
        self.mean_file = '../data/imagenet_mean.npy'

        ### Setup session
        log('setup', 'Launching session')
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        ### Construct network structure
        log('setup', 'Creating network')
        with tf.device(self.device):
            ### Setup inputs
            self.test_img = tf.placeholder(tf.float32, 
                [self.batch_size, 256, 256, 3])
            ### Construct CNN
            self.cnn = Alexnet(self.model_path, self.hash_bit_num, self.sess)
            ### Construct train net
            test_img = self.preprocess_img(self.test_img, self.batch_size, False)
            with tf.variable_scope("cnn"):
                feature = self.cnn.extract(test_img)
                tf.get_variable_scope().reuse_variables()
                self.hash_bit = tf.tanh(feature)
                tf.get_variable_scope().reuse_variables()

            ### init all variables
            log('setup', 'Initialize all variables')
            #self.sess.run(tf.global_variables_initializer())

    def test(self, test_set):
        log('test', 'testing starts')
        n_samples = test_set._n_samples

        # modified by caozhangjie
        #codes = np.empty((0, self.hash_bit_num))
        #labels = np.empty((0, self.n_class))
        codes = []
        labels = []

        samples = 0
        while 1:
            test_img, test_label, has_next = test_set.next_batch(self.batch_size)
            hash_bit = self.sess.run(self.hash_bit, feed_dict={self.test_img: test_img})
            #print hash_bit
            if has_next:
                samples += self.batch_size

                # modified by caozhangjie
                #codes = np.append(codes, hash_bit, axis = 0)
                #labels = np.append(labels, test_label, axis = 0)
                codes.append(hash_bit)
                labels.append(test_label)

            else:
                rest_num = n_samples - samples

                # modified by caozhangjie
                #codes = np.append(codes, hash_bit[0:rest_num, :], axis = 0)
                #labels = np.append(labels, test_label[0:rest_num, :], axis = 0)
                codes.append(hash_bit[0:rest_num, :])
                labels.append(test_label[0:rest_num, :])

                print 'done!'
                break
            print str(samples) + '/' + str(n_samples)
        codes = np.vstack(codes)
        labels = np.vstack(labels)
        assert codes.shape[0] == n_samples, "sample num error"
        assert labels.shape[0] == n_samples, "sample num error"
        return codes, labels


    def preprocess_img(self, img, batch_size, train_phase, oversample=False):
        '''
        pre-process input image:
        Args:
            img: 4-D tensor
            batch_size: Int 
            train_phase: Bool
        Return:
            distorted_img: 4-D tensor
        '''
        reshaped_image = tf.cast(img, tf.float32)
        mean = tf.constant(np.load(self.mean_file), dtype=tf.float32, shape=[1, 256, 256, 3])
        reshaped_image -= mean
        crop_height = IMAGE_SIZE
        crop_width = IMAGE_SIZE
        if train_phase:
            distorted_img = tf.stack([tf.random_crop(tf.image.random_flip_left_right(each_image), [crop_height, crop_width, 3]) for each_image in tf.unstack(reshaped_image)])
        else:
            if oversample:
                distorted_img1 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img2 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img3 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img4 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img5 = tf.stack([tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 14, 14, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img6 = tf.stack([tf.image.crop_to_bounding_box(each_image, 0, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img7 = tf.stack([tf.image.crop_to_bounding_box(each_image, 28, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img8 = tf.stack([tf.image.crop_to_bounding_box(each_image, 28, 0, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img9 = tf.stack([tf.image.crop_to_bounding_box(each_image, 0, 28, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
                distorted_img0 = tf.stack([tf.image.crop_to_bounding_box(each_image, 14, 14, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])            
                distorted_img = tf.concat(0, [distorted_img1, distorted_img2, distorted_img3, distorted_img4, distorted_img5, distorted_img6, distorted_img7, distorted_img8, distorted_img9, distorted_img0])
            else:
                distorted_img = tf.stack([tf.image.crop_to_bounding_box(each_image, 14, 14, crop_height, crop_width) for each_image in tf.unstack(reshaped_image)])
        return distorted_img


def test(config):
    params = {}
    net = Net(config)
    test_set = Dataset(config['test_list'], config['n_class'])
    begin = datetime.datetime.now()
    params['validation_code'], params['validation_labels'] = net.test(test_set)
    end = datetime.datetime.now()
    print end-begin
    database_set = Dataset(config['database_list'], config['n_class'])
    params['database_code'], params['database_labels'] = net.test(database_set)
    return params

config = dict(
    device = '/gpu:' + sys.argv[1],
    batch_size = 64,
    n_class = 12,
    hash_bit_num = 48,
    model_path = "../models/dhn_ad_0.1_10000/model.ckpt",
    #test_list = "../data/parallel/all_test.txt",
    #database_list = "../data/parallel/all_database.txt"
    test_list = "../data/challenge/validation_test_list.txt",
    database_list = "../data/challenge/validation_database_list.txt"
)

if __name__ == "__main__":
    generate_code = True
    if generate_code:
        params = test(config)
        params["path"] = "saving_code"
        params["prefix"] = "our_ad2_0.1"
        save_code_and_label(params)
    else:
        params = {}
        params["path"] = "saving_code"
        params["prefix"] = "our_ad2_0.1"
        load_code_and_label(params)
    params['R'] = 5000
    mAP = mean_average_precision(params)
    print mAP
