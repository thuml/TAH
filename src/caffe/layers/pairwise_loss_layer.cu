#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ForwardGPU(const int nthreads, const int num, const Dtype* similarity, 
       const Dtype* exp_product, const Dtype* product, const Dtype threshold, Dtype* count, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      count[index] = Dtype(1.0);
      if((threshold >= 0) && (product[index] >= threshold)){
          loss_data[index] = product[index] * (1 - (similarity[index] > 0));
      }
      else{
          loss_data[index] = log(1 + exp_product[index]) - (similarity[index] > 0) * product[index];
      }
  }
}

template <typename Dtype>
__global__ void TSNEProduct(const int nthreads, const int outer_num, const int inner_num, const Dtype* input1, const Dtype* input2, Dtype* tproduct, Dtype* out){
    CUDA_KERNEL_LOOP(index, nthreads){
        int data_id1 = index / outer_num;
        int data_id2 = index % outer_num;
        Dtype sum = 0;
        for (int i = 0; i < inner_num; i++){
            sum += (input1[data_id1 * inner_num + i] - input2[data_id2 * inner_num + i])*(input1[data_id1 * inner_num + i] - input2[data_id2 * inner_num + i]);
        }
        tproduct[index] = sum;
        out[index] = (inner_num / 2) / (1.0 + sum);
    }
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* similarity = pairwise_sim_.mutable_gpu_data();
  Dtype* dot_product = pairwise_sim_.mutable_gpu_diff();
  Dtype* exp_product = loss_.mutable_gpu_diff();
  Dtype* loss_data = loss_.mutable_gpu_data();
  Dtype* count = temp_.mutable_gpu_data();
  Dtype* label = bottom[1]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int nthreads = outer_num_ * outer_num_;
  //calculate similarity matrix according to label
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
          Dtype(1.0), label, label, Dtype(0.0), similarity);
  Dtype* tproduct = tsne_.mutable_gpu_data();
  TSNEProduct<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, inner_num_, bottom_data, bottom_data, tproduct, dot_product);
 
  caffe_gpu_exp(outer_num_ * outer_num_, dot_product, exp_product);
  
  //calculate pairwise loss
  ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, similarity, exp_product, 
              dot_product, threshold_, count, loss_data);
  
  Dtype loss, count_num;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  caffe_gpu_asum(nthreads, count, &count_num);
  loss /= (count_num > 0 ? count_num : Dtype(1));
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void BackwardGPU(const int nthreads, const int outer_num, const int inner_num,
          const Dtype* similarity, const Dtype* exp_product,const Dtype* tproduct,const Dtype threshold,  Dtype* count, Dtype* diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      if ((inner_num / 2)/(1. + tproduct[index])>threshold) {
        diff[index] = 1.0 * (1- (similarity[index] > 0));
      }
      else {
        diff[index] = (
          1 / (1 + 1 / exp_product[index]) - 
          (similarity[index] > 0));
      }
      diff[index] /= Dtype(outer_num * outer_num);
      count[index] = Dtype(1.0);
   }
}

template <typename Dtype>
__global__ void TSNEBackward31(const int nthreads, const Dtype scale, const int outer_num, const int inner_num, const Dtype* top_diff, const Dtype* tproduct,const Dtype* similarity, const Dtype threshold,  const Dtype* input, Dtype* diff){
    CUDA_KERNEL_LOOP(index, nthreads){  
      for (int i = 0; i < outer_num; i++) {
        Dtype temp_diff = top_diff[int(index / inner_num) * outer_num + i] * (-(inner_num / 2) / ((1.0+tproduct[int(index / inner_num) * outer_num + i])*(1.0+tproduct[int(index / inner_num) * outer_num + i])));
      diff[index] +=  temp_diff * 2.0 * (input[index] - input[i * inner_num + index % inner_num]);
      }
    }
}

template <typename Dtype>
__global__ void TSNEBackward32(const int nthreads, const Dtype scale, const int outer_num, const int inner_num, const Dtype* top_diff, const Dtype* tproduct, const Dtype* similarity, const Dtype threshold, const Dtype* input, Dtype* diff){
    CUDA_KERNEL_LOOP(index, nthreads){
      for (int i = 0; i < outer_num; i++){
        Dtype temp_diff = top_diff[i * outer_num + index / inner_num] * (-(inner_num / 2) / ((1.0+tproduct[i * outer_num + index / inner_num])*(1.0+tproduct[ i * outer_num + index / inner_num])));
        diff[index]  += temp_diff * 2.0 * (input[index] - input[i * inner_num + index % inner_num]);
      }
   }
}


template <typename Dtype>
__global__ void CleanBlob(const int nthreads, Dtype* clean_blob)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        clean_blob[index] = Dtype(0);
    }
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* diff = temp_.mutable_gpu_data();
    Dtype* count = temp_.mutable_gpu_diff();
    const Dtype* similarity = pairwise_sim_.gpu_data();
    const Dtype* exp_product = loss_.gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    int nthreads = outer_num_ * outer_num_;
  
    //calculate diff
    BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, inner_num_, similarity,
                exp_product,tsne_.gpu_data(),threshold_, count, diff);
        
    //copy to bottom_diff
    Dtype count_num;
    caffe_gpu_asum(nthreads, count, &count_num);
    const Dtype* tproduct = tsne_.gpu_data();
    TSNEBackward31<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_ * inner_num_, loss_weight_/ (count_num > 0? count_num : Dtype(1)), outer_num_, inner_num_, diff, tproduct, similarity, threshold_,  bottom_data, bottom_diff);
    TSNEBackward32<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_ * inner_num_, loss_weight_/ (count_num > 0? count_num : Dtype(1)), outer_num_, inner_num_, diff, tproduct, similarity, threshold_, bottom_data, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseLossLayer);

}  // namespace caffe
