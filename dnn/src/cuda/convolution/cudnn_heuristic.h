/**
 * \file dnn/src/cuda/convolution/cudnn_heuristic.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/cuda/convolution/backward_data/algo.h"
#include "src/cuda/convolution/backward_filter/algo.h"

namespace megdnn {
namespace cuda {
namespace convolution {

enum class ConvolutionType {
    FORWARD = 0,
    BACKWARD_FILTER = 1,
    BACKWARD_DATA = 2
};

bool heuristic_params_available(
        int cuda_major, int cuda_minor, size_t* layer_num_p,
        const size_t** layers_dim_p, const float** matrices_p,
        const float** biases_p, const float** alpha_p, const float** beta_p,
        const ConvolutionType& conv_type, float** hidden_units_p,
        float** time_pred_p, float** mask_p);

class PerformanceModelBase {
public:
    static float element_ReLU(float element) {
        return element > 0.0 ? element : 0.0;
    }
    static bool predict_time_success(const TensorLayout* x_layout,
                                     const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& filter,
                                     const ConvolutionType& conv_type,
                                     float** mask_p, float** time_pred_p,
                                     size_t* output_dim_p);

private:
    static bool args_is_proper(
            const TensorLayout* x_layout,
            const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& filter);
    static void predict_time(const size_t layer_num, const size_t* layers_dim,
                             const size_t* input_params, const float* matrices,
                             const float* biases, const float* alpha,
                             const float* beta, float* hidden_units,
                             float* time_pred);
};

class PerformanceModelBackwardFilter : public PerformanceModelBase {
public:
    static bool get_algo_backward_filter_success(
            const ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs& args,
            const CUDNNBwdFilterDescs& D, const size_t workspace_limit_in_bytes,
            cudnnConvolutionBwdFilterAlgo_t* algo);

private:
    static void gen_mask_backward_filter(
            float* mask, const size_t output_dim,
            const ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs& args,
            const CUDNNBwdFilterDescs& D,
            const size_t workspace_limit_in_bytes);
};

class PerformanceModelBackwardData : public PerformanceModelBase {
public:
    static bool get_algo_backward_data_success(
            const ConvolutionBackwardDataImpl::AlgoBase::SizeArgs& args,
            const CUDNNBwdDataDescs& D, const size_t workspace_limit_in_bytes,
            cudnnConvolutionBwdDataAlgo_t* algo);

private:
    static void gen_mask_backward_data(
            float* mask, const size_t output_dim,
            const ConvolutionBackwardDataImpl::AlgoBase::SizeArgs& args,
            const CUDNNBwdDataDescs& D, const size_t workspace_limit_in_bytes);
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn
