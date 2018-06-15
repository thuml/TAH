# TAH
Caffe implementation for ["Transfer Adversarial Hashing for Hamming Space Retrieval" (AAAI 2018)](https://arxiv.org/abs/1712.04616) 

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

Caffe

Python 2.7

## Modification on Caffe
- Add "MultiLabelData" layer to process multi-label dataset.
- Add "PairwiseLoss" layer implementing the t-distribution pairwise loss described in our paper.
- Add "GradientScaler" layer to inverse and scale the gradient for adversarial learning.

## Datasets
We use NUS-WIDE and Visda datasets in our experiments. You can download the NUS-WIDE dataset [here](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing).
As for Visda dataset, we use the recognition part of Visual Domain Adaptation Challenge dataset on TASK-CV 2017 Workshop, which can be downloaded [here](https://github.com/VisionLearningGroup/taskcv-2017-public). 

You can also modify the list file(txt format) in ./data as you like. Each line in the list file follows the following format:
```
<image path><space><one hot label representation>
```
## Compiling
The compiling process is the same as caffe. You can refer to Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html).

## Training
First, you need to download the AlexNet pre-trained model on ImageNet from [here](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) and move it to [./models/bvlc_reference_caffenet](./models/bvlc_reference_caffenet).
Then, you can train the model for each dataset using the followling command.
```
dataset_name = imagenet, nuswide_81 or coco
./build/tools/caffe train -solver models/train/dataset_name/solver.prototxt -weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu gpu_id
```

## Evaluation
You can evaluate the Mean Average Precision(MAP) result on each dataset using the followling command.
```
python models/predict/predict.py --gpu_id 1,2,3 --nthreads 4 --database_path data/visda/real_database_list.txt --test_path data/visda/real_test_list.txt --code_path real --snapshot_path models/train/caffemodel/synthetic_48bit.caffemodel
```
We provide some trained models for each dataset for each code length in our experiment for evaluation. You can download them [here](https://drive.google.com/drive/folders/1q845_lozfiepTfGmOOjR46YxkQXISGcK?usp=sharing) if you want to use them.

If you have generated the hash code by the previous step or by other method and want to test the MAP of the hash code. You can specify the code_path parameter.
```
python models/predict/predict.py --code_path real --load_code True
```

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{DBLP:conf/aaai/CaoLH018,
  author    = {Zhangjie Cao and
               Mingsheng Long and
               Chao Huang and
               Jianmin Wang},
  title     = {Transfer Adversarial Hashing for Hamming Space Retrieval},
  booktitle = {Proceedings of the Thirty-Second {AAAI} Conference on Artificial Intelligence,
               New Orleans, Louisiana, USA, February 2-7, 2018},
  year      = {2018},
  url       = {https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17256},
  timestamp = {Thu, 03 May 2018 17:03:19 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/aaai/CaoLH018},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Contact
If you have any problem about our code, feel free to contact caozhangjie14@gmail.com or describe your problem in Issues.
