##################################################################################
# 2017 01.16 Created by Shichen Liu                                              #
# Residual Transfer Network implemented by tensorflow                            #
#                                                                                #
#                                                                                #
##################################################################################

import os
import sys
import numpy as np
import scipy.io as sio
import scipy as sp
import random
import cv2
import caffe
from multiprocessing import Pool

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
    
def get_codes_and_labels(params):
    caffe.set_device(params['gpu_id'])
    caffe.set_mode_gpu()
    model_file = params['model_file']
    pretrained_model = params['pretrained_model']
    dims = params['image_dims']
    scale = params['scale']
    database = open(params['database'], 'r').readlines()
    validation = open(params['validation'], 'r').readlines()
    batch_size = params['batch_size']

    if 'mean_file' in params:
        mean_file = params['mean_file']
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, mean=np.load(mean_file).mean(        1).mean(1), raw_scale=scale)
    else:
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, raw_scale=scale)

    database_code = []
    validation_code = []
    database_labels = []
    validation_labels = []
    cur_pos = 0

    while 1:
        lines = database[cur_pos : cur_pos + batch_size]
        if len(lines) == 0:
            break;
        cur_pos = cur_pos + len(lines)
        images = [caffe.io.load_image(line.strip().split(" ")[0]) for line in lines]
        labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]
        codes = net.predict(images, oversample=False)
        [database_code.append(c) for c in codes]
        [database_labels.append(l) for l in labels]

        print str(cur_pos) + "/" + str(len(database))
        if len(lines) < batch_size:
            break;

    cur_pos = 0
    while 1:
        lines = validation[cur_pos : cur_pos + batch_size]
        if len(lines) == 0:
            break;
        cur_pos = cur_pos + len(lines)
        images = [caffe.io.load_image(line.strip().split(" ")[0]) for line in lines]
        labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]

        codes = net.predict(images, oversample=False)
        [validation_code.append(c) for c in codes]
        [validation_labels.append(l) for l in labels]

        print str(cur_pos) + "/" + str(len(validation))
        if len(lines) < batch_size:
            break;

    return dict(database_code=database_code, database_labels=database_labels, validation_code=validation_code, validation_labels=validation_labels)
 

def load_code_and_label(params):
    path = params['path']
    params['database_code'] = np.load(opath.join(path, params["prefix"]+"database_code.npy"))
    params['database_labels'] = np.load(opath.join(path, params["prefix"]+"database_label.npy"))
    params['validation_code'] = np.load(opath.join(path, params["prefix"]+"validation_code.npy"))
    params['validation_labels'] = np.load(opath.join(path, params["prefix"]+"validation_label.npy"))

def precision_recall(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(validation_code, database_code.T)
    ground_truth = np.dot(validation_labels, database_labels.T)
    count_ = np.array(sim)
    ground_truth[ground_truth>0] = 1.0
    count_all = np.sum(ground_truth, axis=1)
    length_of_code = database_code.shape[1]
    radius = length_of_code
    precision = []
    recall = []
    for i in xrange(length_of_code):
        radius = radius - 2
        radius_find = sim > radius
        not_radius_find = sim <= radius
        count_[radius_find] = 1.0
        count_[not_radius_find] = 0.0
        g_truth = np.array(ground_truth)
        g_truth[not_radius_find] = 0.0
        count_1 = np.sum(count_, axis=1)
        for i in xrange(count_1.shape[0]):
            if count_1[i] == 0:
                count_1[i] = 1.0
        precision.append(np.mean(np.divide(np.sum(g_truth, axis=1), count_1)))
        recall.append(np.mean(np.divide(np.sum(g_truth, axis=1), count_all)))
    return precision, recall

def precision_recall_hamming2(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    code_length = database_code.shape[1]
    sim_hamming2 = sim >= code_length - 4.
    hamming2_num = np.sum(sim_hamming2, axis=0)
    query_interp_points = np.array(range(101)) / 100.0
    all_p = []
    for i in range(query_num):
      if hamming2_num[i] > 0:
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:int(hamming2_num[i])], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        if relevant_num != 0:
            Lx = np.cumsum(imatch)
            P = np.divide(Lx.astype(float), np.arange(1, int(hamming2_num[i])+1, 1))
            R = Lx.astype(float) / relevant_num
            y = np.interp(query_interp_points, R, P)
            all_p.append(y)
        else:
            Lx = np.cumsum(imatch)
            P = np.divide(Lx.astype(float), np.arange(1, int(hamming2_num[i])+1, 1))
            R = np.ones(Lx.astype(float).shape).astype(float)
            y = np.interp(query_interp_points, R, P)
            all_p.append(y)
      else:
        y = np.interp(query_interp_points, np.array([0., 1.]), np.array([0., 0.]))
        all_p.append(y)
    P = np.mean(np.array(all_p), axis=0)
    
    return [query_interp_points, P]

nthreads = 4
ndevices = 3
params = []

for gpu_id in range(ndevices):
    for i in range(nthreads):
        params.append(dict(model_file="./deploy2.prototxt",
                      pretrained_model="../nus_wide/nus_step_400_lr_4e-5_qw_0_8_iter_10000.caffemodel",
                      image_dims=(256,256),
                      scale=255,
                      database="../../../data/nus_wide2/parallel/database" + str(nthreads*gpu_id+i) + ".txt",
                      validation="../../../data/nus_wide2/parallel/test" + str(nthreads*gpu_id+i) + ".txt",
                      batch_size=50,
                      mean_file="./ilsvrc_2012_mean.npy",
                      gpu_id=gpu_id))

pool = Pool(nthreads*ndevices)
results = pool.map(get_codes_and_labels, params)

code_and_label = results[0]
for i in range(1, nthreads*ndevices):
    [code_and_label['database_code'].append(c) for c in results[i]['database_code']]
    [code_and_label['database_labels'].append(c) for c in results[i]['database_labels']]
    [code_and_label['validation_code'].append(c) for c in results[i]['validation_code']]
    [code_and_label['validation_labels'].append(c) for c in results[i]['validation_labels']]

code_and_label['R'] = 5000
mAP,recall = precision_recall_hamming2(code_and_label)

aaa = open('./result', 'w')
aaa.write(str(recall))
print recall
