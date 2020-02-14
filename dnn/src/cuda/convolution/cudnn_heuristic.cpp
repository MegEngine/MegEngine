/**
 * \file dnn/src/cuda/convolution/cudnn_heuristic.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./cudnn_heuristic.h"
#include "megdnn.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

bool convolution::PerformanceModelBase::args_is_proper(
        const TensorLayout* x_layout,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& filter) {
    bool available = (x_layout->dtype == dtype::Float32() &&
                      filter.format == param::Convolution::Format::NCHW &&
                      filter.should_flip == 0 && filter.stride[0] == 1 &&
                      filter.stride[1] == 1 && filter.spatial_ndim == 2 &&
                      filter.dilation[0] == 1 && filter.dilation[1] == 1);
    return available;
}

bool convolution::PerformanceModelBase::predict_time_success(
        const TensorLayout* x_layout, const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& filter,
        const ConvolutionType& conv_type, float** mask_p, float** time_pred_p,
        size_t* output_dim_p) {
    size_t layer_num;
    const size_t* layers_dim;
    size_t input_params[9];
    const float* matrices;
    const float* biases;
    const float* alpha;
    const float* beta;
    float* hidden_units;

    if (!(args_is_proper(x_layout, filter))) {
        return false;
    }

    if (!convolution::heuristic_params_available(
                cuda::current_device_prop().major,
                cuda::current_device_prop().minor, &layer_num, &layers_dim,
                &matrices, &biases, &alpha, &beta, conv_type, &hidden_units,
                time_pred_p, mask_p)) {
        return false;
    }

    input_params[0] = x_layout->shape[0];
    input_params[1] = x_layout->shape[1];
    input_params[2] = x_layout->shape[2];
    input_params[3] = x_layout->shape[3];
    input_params[4] = filter.ocpg;
    input_params[5] = filter.spatial[0];
    input_params[6] = filter.spatial[1];
    input_params[7] = filter.padding[0];
    input_params[8] = filter.padding[1];

    predict_time(layer_num, layers_dim, input_params, matrices, biases, alpha,
                 beta, hidden_units, *time_pred_p);

    *output_dim_p = layers_dim[layer_num - 1];

    return true;
}

void convolution::PerformanceModelBase::predict_time(
        const size_t layer_num, const size_t* layers_dim,
        const size_t* input_params, const float* matrices, const float* biases,
        const float* alpha, const float* beta, float* hidden_units,
        float* time_pred) {
    size_t layer_ind;
    size_t i, j;
    const float *matrix_entry = matrices, *bias_entry = biases;
    float *prev_entry, *next_entry = hidden_units;
    size_t shape;

    for (j = 0; j < layers_dim[1]; ++j) {
        for (i = 0; i < layers_dim[0]; ++i) {
            next_entry[j] +=
                    matrix_entry[j * layers_dim[0] + i] * input_params[i];
        }
        next_entry[j] += bias_entry[j];
        next_entry[j] = element_ReLU(next_entry[j]);
    }
    prev_entry = next_entry;
    next_entry += layers_dim[1];
    matrix_entry += layers_dim[0] * layers_dim[1];
    bias_entry += layers_dim[1];

    for (layer_ind = 1; layer_ind < layer_num - 2; ++layer_ind) {
        for (j = 0; j < layers_dim[layer_ind + 1]; ++j) {
            for (i = 0; i < layers_dim[layer_ind]; ++i) {
                next_entry[j] += matrix_entry[j * layers_dim[layer_ind] + i] *
                                 prev_entry[i];
            }
            next_entry[j] += bias_entry[j];
            next_entry[j] = element_ReLU(next_entry[j]);
        }
        prev_entry = next_entry;
        next_entry += layers_dim[layer_ind + 1];
        matrix_entry += layers_dim[layer_ind] * layers_dim[layer_ind + 1];
        bias_entry += layers_dim[layer_ind + 1];
    }

    for (j = 0; j < layers_dim[layer_num - 2]; ++j) {
        for (i = 0; i < layers_dim[layer_num - 1]; ++i) {
            time_pred[j] += matrix_entry[j * layers_dim[i]] * input_params[i];
        }
        time_pred[j] += bias_entry[j];
    }

    shape = input_params[0] * input_params[1] * input_params[4] *
            (input_params[2] + input_params[7] * 2 - input_params[5] + 1) *
            (input_params[3] + input_params[8] * 2 - input_params[6] + 1) *
            input_params[5] * input_params[6];
    for (i = 0; i < layers_dim[layer_num - 1]; ++i) {
        time_pred[i] = std::exp2f(time_pred[i] * beta[i]) * (shape / alpha[i]);
    }
}

/* backward filter */
void convolution::PerformanceModelBackwardFilter::gen_mask_backward_filter(
        float* mask, const size_t output_dim,
        const ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs& args,
        const CUDNNBwdFilterDescs& D,
        const size_t workspace_size_limit_in_bytes) {
    size_t i;
    size_t workspace_size;
    for (i = 0; i < output_dim; ++i) {
        mask[i] = -1.0f;
        auto cudnnStat = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                args.handle->cudnn_handle(), D.src_desc.desc, D.diff_desc.desc,
                D.conv_desc.desc, D.grad_desc.desc,
                static_cast<cudnnConvolutionBwdFilterAlgo_t>(i),
                &workspace_size);
        if (cudnnStat == CUDNN_STATUS_SUCCESS &&
            workspace_size < workspace_size_limit_in_bytes) {
            mask[i] = 1.0f;
        }
    }
}

bool convolution::PerformanceModelBackwardFilter::
        get_algo_backward_filter_success(
                const ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs& args,
                const CUDNNBwdFilterDescs& D,
                const size_t workspace_limit_in_bytes,
                cudnnConvolutionBwdFilterAlgo_t* algo) {
    float* mask;
    size_t output_dim;
    float* time_pred;

    if (!predict_time_success(args.src_layout, args.grad_filter_meta,
                              ConvolutionType::BACKWARD_FILTER, &(mask),
                              &(time_pred), &(output_dim))) {
        return false;
    }

    gen_mask_backward_filter(mask, output_dim, args, D,
                             workspace_limit_in_bytes);

    size_t i, selected = 0;
    for (i = 0; i < output_dim; ++i) {
        if (mask[i] > 0 && time_pred[i] < time_pred[selected]) {
            selected = i;
        }
    }
    *algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(selected);

    return mask[selected] > 0;
}

/* backward data */
void convolution::PerformanceModelBackwardData::gen_mask_backward_data(
        float* mask, const size_t output_dim,
        const ConvolutionBackwardDataImpl::AlgoBase::SizeArgs& args,
        const CUDNNBwdDataDescs& D,
        const size_t workspace_size_limit_in_bytes) {
    size_t i;
    size_t workspace_size;
    for (i = 0; i < output_dim; ++i) {
        mask[i] = -1.0f;
        auto cudnnStat = cudnnGetConvolutionBackwardDataWorkspaceSize(
                args.handle->cudnn_handle(), D.filter_desc.desc,
                D.diff_desc.desc, D.conv_desc.desc, D.grad_desc.desc,
                static_cast<cudnnConvolutionBwdDataAlgo_t>(i), &workspace_size);
        if (cudnnStat == CUDNN_STATUS_SUCCESS &&
            workspace_size < workspace_size_limit_in_bytes) {
            mask[i] = 1.0f;
        }
    }
}

bool convolution::PerformanceModelBackwardData::get_algo_backward_data_success(
        const ConvolutionBackwardDataImpl::AlgoBase::SizeArgs& args,
        const CUDNNBwdDataDescs& D, const size_t workspace_limit_in_bytes,
        cudnnConvolutionBwdDataAlgo_t* algo) {
    float* mask;
    size_t output_dim;
    float* time_pred;

    if (!predict_time_success(args.grad_layout, args.filter_meta,
                              ConvolutionType::BACKWARD_DATA, &mask, &time_pred,
                              &output_dim)) {
        return false;
    }

    gen_mask_backward_data(mask, output_dim, args, D, workspace_limit_in_bytes);

    size_t i, selected = 0;
    for (i = 0; i < output_dim; ++i) {
        if (mask[i] > 0 && time_pred[i] < time_pred[selected]) {
            selected = i;
        }
    }

    // special case:
    // if the filter shape in cudnnConvolutionBackwardData is too asymmetric,
    // the performance of algo1 is dramatically reduced,
    // we temporarily choose algo0.
    if (args.filter_meta.spatial[0] / args.filter_meta.spatial[1] > 32 ||
        args.filter_meta.spatial[1] / args.filter_meta.spatial[0] > 32) {
        selected = 0;
    }
    *algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(selected);

    return mask[selected] > 0;
}
