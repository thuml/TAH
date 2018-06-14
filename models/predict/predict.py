import numpy as np
import scipy as sp
import sys
import caffe
from multiprocessing import Pool
import argparse
import os
import os.path as osp

def save_code_and_label(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_label = np.array(params['database_label'])
    validation_label = np.array(params['validation_label'])
    path = params['path']
    np.save(osp.join(path, "database_code.npy"), database_code)
    np.save(osp.join(path, "database_label.npy"), database_label)
    np.save(osp.join(path, "validation_code.npy"), validation_code)
    np.save(osp.join(path, "validation_label.npy"), validation_label)

def load_code_and_label(path):
    params = {}
    params['database_code'] = np.load(osp.join(path, "database_code.npy"))
    params['database_label'] = np.load(osp.join(path, "database_label.npy"))
    params['validation_code'] = np.load(osp.join(path, "validation_code.npy"))
    params['validation_label'] = np.load(osp.join(path, "validation_label.npy"))
    return params

def top_k(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_label = np.array(params['database_label'])
    validation_label = np.array(params['validation_label'])
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(validation_code, database_code.T)
    ranking = np.argsort(-sim, axis=1)
    ground_truth = np.dot(validation_label, database_label.T)
    ground_truth[ground_truth>0] = 1.0
    ranking_k = ranking < params["k"]
    not_ranking_k = ranking >= params["k"]
    ground_truth[not_ranking_k] = 0.0
    return (np.mean(np.sum(ground_truth, axis=1)/float(params["k"])), ground_truth.shape)

def mean_average_precision(params):
    database_code = params['database_code']
    validation_code = params['validation_code']
    database_label = params['database_label']
    validation_label = params['validation_label']
    R = params['R']
    query_num = validation_code.shape[0]
    
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    
    for i in range(query_num):
        label = validation_label[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_label[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        else:
            APX.append(0)
    
    return np.mean(np.array(APx))

def mean_average_precision_hamming2(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_label = np.array(params['database_label'])
    validation_label = np.array(params['validation_label'])
    query_num = validation_code.shape[0]
    
    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    code_length = database_code.shape[1]
    sim_hamming2 = sim >= code_length - 4.
    hamming2_num = np.sum(sim_hamming2, axis=0)
    
    for i in range(query_num):
      if hamming2_num[i] > 0:
        label = validation_label[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_label[idx[0:int(hamming2_num[i])], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, int(hamming2_num[i])+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        else:
            APx.append(0)
      else:
        APx.append(0)
    
    return np.mean(np.array(APx))       

def hamming_radius2(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_label = np.array(params['database_label'])
    validation_label = np.array(params['validation_label'])
    R = params['R']
    query_num = validation_code.shape[0]
    
    sim = np.dot(validation_code, database_code.T)
    ground_truth = np.dot(validation_label, database_label.T)
    ground_truth[ground_truth>0] = 1.0 
    length_of_code = database_code.shape[1]
    radius2 = sim >= (length_of_code - 4)
    not_radius2 = sim < (length_of_code - 4)
    sim[radius2] = 1.0
    sim[not_radius2] = 0.0
    ground_truth[not_radius2] = 0.0
    count_radius2 = np.sum(sim, axis=1)
    for i in xrange(count_radius2.shape[0]):
        if count_radius2[i] == 0:
            count_radius2[i] = 1.0
    return np.mean(np.divide(np.sum(ground_truth, axis=1), count_radius2))

def precision_recall(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_label = np.array(params['database_label'])
    validation_label = np.array(params['validation_label'])
    R = params['R']
    query_num = validation_code.shape[0]
    
    sim = np.dot(validation_code, database_code.T)
    ground_truth = np.dot(validation_label, database_label.T)
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
    database_label = np.array(params['database_label'])
    validation_label = np.array(params['validation_label'])
    query_num = validation_code.shape[0]
    
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
        label = validation_label[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_label[idx[0:int(hamming2_num[i])], :] == label, axis=1) > 0
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


def get_codes_and_labels(params):
    caffe.set_device(params['gpu_id'])
    caffe.set_mode_gpu()
    model_file = params['model_file']
    pretrained_model = params['pretrained_model']
    dims = params['image_dims']
    scale = params['scale']
    database = params['database']
    validation = params['validation']
    batch_size = params['batch_size']

    if 'mean_file' in params:
        mean_file = params['mean_file']
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, mean=np.load(mean_file).mean(1).mean(1), raw_scale=scale)
    else:
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, raw_scale=scale)
    
    database_code = []
    validation_code = []
    database_label = []
    validation_label = []
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
        [database_label.append(l) for l in labels]
        
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
        [validation_label.append(l) for l in labels]
        
        print str(cur_pos) + "/" + str(len(validation))
        if len(lines) < batch_size:
            break;
        
    return dict(database_code=np.sign(database_code), database_label=database_label, validation_code=np.sign(validation_code), validation_label=validation_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trasfer Adversarial Hashing')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--database_path', type=str, default='coco', help="database set path")
    parser.add_argument('--test_path', type=str, default=48, help="test set path")
    parser.add_argument('--snapshot_path', type=str, default='ResNet50', help="base network type")
    parser.add_argument('--code_path', type=str, help="save path prefix")
    parser.add_argument('--load_code', type=bool, default=False, help="use existing code or not")
    parser.add_argument('--nthreads', type=int, help="number of threads in each devices")
    args = parser.parse_args()

    if not args.load_code:
        devices = [int(device) for device in args.gpu_id.split(",")]
        nthreads = args.nthreads
        ndevices = len(devices)
        threads_num = nthreads * ndevices
        
        database_path = args.database_path
        test_path = args.test_path
        all_database = open(database_path).readlines()
        all_test = open(test_path).readlines()
        len_database = len(all_database)
        len_test = len(all_test)
        len_s_database = len_database / threads_num
        len_s_test = len_test / threads_num
        database = []
        test = []
        for i in range(threads_num):
            database.append([])
            test.append([])
            database[i] += all_database[i*len_s_database:(i+1)*len_s_database]
            test[i] += all_test[i*len_s_test:(i+1)*len_s_test]
        
        params = []
        for gpu_id in range(ndevices):
            for i in range(nthreads):
                params.append(dict(model_file="./models/predict/deploy.prototxt",
                              pretrained_model=args.snapshot_path,
                              image_dims=(256,256),
                              scale=255,
                              database=database[gpu_id*nthreads+i],
                              validation=test[gpu_id*nthreads+i],
                              batch_size=50,
                              mean_file="./python/caffe/imagenet/ilsvrc_2012_mean.npy",
                              gpu_id=devices[gpu_id]))
        
        pool = Pool(nthreads*ndevices)
        results = pool.map(get_codes_and_labels, params)
        
        code_and_label = results[0]
        for i in range(1, nthreads*ndevices):
            [code_and_label['database_code'].append(c) for c in results[i]['database_code']]
            [code_and_label['database_label'].append(c) for c in results[i]['database_label']]
            [code_and_label['validation_code'].append(c) for c in results[i]['validation_code']]
            [code_and_label['validation_label'].append(c) for c in results[i]['validation_label']]
        code_and_label['path'] = "./data/code/"+args.code_path
        if not osp.exists(code_and_label['path']):
            os.mkdir(code_and_label['path'])
        save_code_and_label(code_and_label)
    else:
        code_and_label = load_code_and_label("./data/code/"+args.code_path)

    code_and_label['R'] = 5000
    mAP = mean_average_precision_hamming2(code_and_label)
    print mAP
