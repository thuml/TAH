import numpy as np
import scipy as sp
import sys
import caffe
from multiprocessing import Pool 
import h5py
import argparse

def save_code_and_label(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    path = params['path']
    np.save(path + "database_code.npy", database_code)
    np.save(path + "database_label.npy", database_labels)
    np.save(path + "validation_code.npy", validation_code)
    np.save(path + "validation_label.npy", validation_labels)

def mean_average_precision(params):
    database_code = np.sign(params['database_code'])
    validation_code = np.sign(params['validation_code'])
    database_labels = params['database_labels']
    validation_labels = params['validation_labels']
    R = params['R']
    query_num = validation_code.shape[0]

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
        
def get_codes_and_labels(params):
    device_id = params["gpu"]
    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    model_file = params['model_file']
    pretrained_model = params['pretrained_model']
    batch_size = params['batch_size']
    net = caffe.Net(model_file, pretrained_model, caffe.TEST)
    h5test_file = h5py.File(params['validation'], "r")
    h5train_file = h5py.File(params['database'], "r")
    database = h5train_file["database_coco"].value
    validation = h5test_file["val_coco"].value
    database_labels = h5train_file["label_coco"].value
    validation_labels = h5test_file["label_coco"].value
    database_code = []
    validation_code = []
    cur_pos = 0
    
    while 1:
        lines = database[cur_pos : cur_pos + batch_size, :]
        length_t = len(lines)
        if length_t == 0:
            break;
        if length_t < batch_size:
            lines = np.vstack([lines, np.zeros(((batch_size - length_t), int(lines.shape[1])))])
        cur_pos = cur_pos + length_t
        net.blobs["data"] = np.reshape(lines, [batch_size, lines.shape[1], 1, 1])
        codes = net.forward()
        codes = np.array(codes["feature3"])
        if length_t < batch_size:
            temp_code = database_code[(-batch_size):(length_t - batch_size)]
            database_code += temp_code
        else:
            [database_code.append(c) for c in codes]
        print str(cur_pos) + "/" + str(len(database))
        if length_t < batch_size:
            break;

    cur_pos = 0
    while 1:
        lines = validation[cur_pos : cur_pos + batch_size, :]
        length_t = len(lines)
        if length_t == 0:
            break;
        if length_t < batch_size:
            lines = np.vstack([lines, np.zeros(((batch_size - length_t), int(lines.shape[1])))])
        cur_pos = cur_pos + length_t
        net.blobs["data"] = np.reshape(lines, [batch_size, lines.shape[1], 1, 1])
        codes = net.forward()
        codes = np.array(codes["feature3"])
        if length_t < batch_size:
            temp_code = validation_code[(-batch_size):(length_t - batch_size)]
            validation_code += temp_code
        else:
            [validation_code.append(c) for c in codes]
        print str(cur_pos) + "/" + str(len(validation))
        if length_t < batch_size:
            break;
    return dict(database_code=np.array(database_code), database_labels=database_labels, validation_code=np.array(validation_code), validation_labels=validation_labels)

parser = argparse.ArgumentParser(description="predict text code")
parser.add_argument("--gpu", dest="gpu", nargs="?", default=0, type=int, help="gpu id")
parser.add_argument("--dataset", dest="dataset", nargs="?", default="coco", type=str, help="dataset name")
parser.add_argument("model_name", nargs="?", default="0", type=str, help="model name")
parser.add_argument("iter_num", nargs="?", default="2000", type=str, help="iter_num")
args = parser.parse_args()

params = dict(model_file="models/thn/adversarial/text_model/deploy.prototxt",
           pretrained_model="models/thn/adversarial/caffemodel/" + args.model_name + "_iter_" + args.iter_num + ".caffemodel",
                  database="../data/coco_nuswide/sklearn_text/database_"+args.dataset+".h5",
                  validation="../data/coco_nuswide/sklearn_text/val_"+args.dataset+".h5",
                  batch_size=50, gpu=args.gpu)

code_and_label = get_codes_and_labels(params)
print "test1"
code_and_label['path'] = "./data/text_model/" + args.model_name
save_code_and_label(code_and_label)
print "test2"
code_and_label['R'] = 5000
mAP = mean_average_precision(code_and_label)

print mAP
