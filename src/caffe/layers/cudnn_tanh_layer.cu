#ifdef USE_CUDNN
#include <vector>
#include <fstream>
#include <stdio.h>
#include "caffe/layers/cudnn_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnActivationForward(this->handle_,
        activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->top_desc_, top_data));
#else
  CUDNN_CHECK(cudnnActivationForward_v4(this->handle_,
        activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->top_desc_, top_data));
#endif
// LOG(INFO)<<"huang chao debug:"<<this->layer_param_.name();
  //caffe_gpu_memcpy(sizeof(Dtype)*top[0]->count(), top[0]->gpu_data(), top[0]->mutable_cpu_data());
 //char filename[100];
 //sprintf(filename, "tanhout0.txt",itercount);
  //itercount++;
  //std::ofstream fout("tanhout0.txt", ios::out);
  //for (int i = 0; i < top[0]->count(); i++)
  //{
  //   fout<<top[0]->cpu_data()[i]<<' ';
  //}
  //fout.close();
}

template <typename Dtype>
void CuDNNTanHLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  //if (this->layer_param_.name() == "tanh8")
  //{
  //caffe_gpu_memcpy(sizeof(Dtype) * top[0]->count(), top[0]->gpu_diff(), top[0]->mutable_cpu_diff());
  //std::ofstream fout("topdiffout1.txt", ios::out);
  //for (int i = 0; i < top[0]->count(); i++)
  //{
  //      fout<<top[0]->cpu_diff()[i]<<' ';
  //}
  //fout.close();
  //}

#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnActivationBackward(this->handle_,
        activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->top_desc_, top_data, this->top_desc_, top_diff,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->bottom_desc_, bottom_diff));
#else
  CUDNN_CHECK(cudnnActivationBackward_v4(this->handle_,
        activ_desc_,
        cudnn::dataType<Dtype>::one,
        this->top_desc_, top_data, this->top_desc_, top_diff,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->bottom_desc_, bottom_diff));
#endif

}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNTanHLayer);

}  // namespace caffe
#endif
