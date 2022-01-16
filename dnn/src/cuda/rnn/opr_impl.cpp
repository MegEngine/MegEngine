/**
 * \file dnn/src/cuda/rnn/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/rnn/opr_impl.h"
#include "src/common/rnn.h"
#include "src/cuda/utils.h"

//#include <cstring>
#include <cudnn.h>
#include <cstdlib>
#include <iostream>

namespace megdnn {
namespace cuda {

using namespace std;

void RNNImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in hx,
        _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
        _megdnn_tensor_out hy, _megdnn_tensor_out reserve_space,
        _megdnn_workspace workspace) {
    Handle* handle = this->handle();

#if false  // CUDNN_MAJOR >= 8
        rnn::RNNForwardDescHolder desc_holder = this->get_desc_holder(input.layout);

        void* workspace_ptr = workspace.raw_ptr;
        void* reserveSpace_ptr = static_cast<uint8_t*>(workspace_ptr) + desc_holder.workspace_size;

        cudnn_check(cudnnRNNForward(
            cudnn_handle(handle), desc_holder.rnn_desc.desc, desc_holder.fwdMode, desc_holder.devSeqLengths,
            desc_holder.x_desc.desc, input.raw_ptr(), desc_holder.y_desc.desc, output.raw_ptr(),
            desc_holder.h_desc.desc, hx.raw_ptr(), hy.raw_ptr(),
            desc_holder.h_desc.desc, nullptr, nullptr,
            desc_holder.weight_size, flatten_weights.raw_ptr(), desc_holder.workspace_size, workspace_ptr,
            desc_holder.reserveSpace_size, reserveSpace_ptr
        ));
#else
    rnn::RNNForwardDescHolder_v6 desc_holder =
            rnn::get_RNNDescHolder_v6(this->handle(), param(), input.layout);
    auto x_desc_arr = rnn::get_descs(desc_holder.x_descs);
    auto y_desc_arr = rnn::get_descs(desc_holder.y_descs);
    RNNWeightFilterDesc w_desc;
    w_desc.set(flatten_weights.layout);

    if (param().fwd_mode == param::RNN::FwdMode::TRAINING) {
        cudnn_check(cudnnRNNForwardTraining(
                cudnn_handle(handle), desc_holder.rnn_desc.desc, desc_holder.seq_len,
                x_desc_arr.data(), input.raw_ptr(), desc_holder.hx_desc.desc,
                hx.raw_ptr(), desc_holder.cx_desc.desc, NULL, w_desc.desc,
                flatten_weights.raw_ptr(), y_desc_arr.data(), output.raw_ptr(),
                desc_holder.hy_desc.desc, hy.raw_ptr(), desc_holder.cy_desc.desc, NULL,
                workspace.raw_ptr, desc_holder.workspace_size, reserve_space.raw_ptr(),
                desc_holder.reserveSpace_size));
    } else {
        cudnn_check(cudnnRNNForwardInference(
                cudnn_handle(handle), desc_holder.rnn_desc.desc, desc_holder.seq_len,
                x_desc_arr.data(), input.raw_ptr(), desc_holder.hx_desc.desc,
                hx.raw_ptr(), desc_holder.cx_desc.desc, nullptr, w_desc.desc,
                flatten_weights.raw_ptr(), y_desc_arr.data(), output.raw_ptr(),
                desc_holder.hy_desc.desc, hy.raw_ptr(), desc_holder.cy_desc.desc,
                nullptr, workspace.raw_ptr, desc_holder.workspace_size));
    }
#endif
}

size_t RNNImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& hx,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout& hy, const TensorLayout& reserve_space) {
#if false  // CUDNN_MAJOR >= 8
        rnn::RNNForwardDescHolder desc_holder = this->get_desc_holder(input);
#else
    rnn::RNNForwardDescHolder_v6 desc_holder =
            rnn::get_RNNDescHolder_v6(this->handle(), param(), input);
#endif
    return desc_holder.workspace_size;
}

size_t RNNImpl::get_reserve_size_in_bytes(const TensorLayout& input) {
    rnn::RNNForwardDescHolder_v6 desc_holder =
            rnn::get_RNNDescHolder_v6(this->handle(), param(), input);
    return desc_holder.reserveSpace_size;
}

/*rnn::RNNForwardDescHolder RNNImpl::get_desc_holder(const TensorLayout& input) {
    Handle* handle = this->handle();
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    size_t input_size = input.shape[2];
    auto _param = param();

    cudnnRNNMode_t mode;
    using NonlineMode = param::RNN::NonlineMode;
    switch (_param.nonlineMode) {
        case NonlineMode::RELU:
            mode = CUDNN_RNN_RELU;
            break;
        case NonlineMode::TANH:
            mode = CUDNN_RNN_TANH;
            break;
    }

    cudnnForwardMode_t fwdMode = CUDNN_FWD_MODE_TRAINING;
    using FwdMode = param::RNN::FwdMode;
    switch (_param.fwd_mode) {
        case FwdMode::TRAINING:
            fwdMode = CUDNN_FWD_MODE_TRAINING;
            break;
        case FwdMode::INFERENCE:
            fwdMode = CUDNN_FWD_MODE_INFERENCE;
            break;
    }

    rnn::RNNForwardDescHolder desc_holder(
            handle, seq_len, batch_size, _param.hidden_size, input_size,
            _param.proj_size, _param.num_layers, _param.bidirectional, _param.bias,
            input.dtype, mode, fwdMode);
    return desc_holder;
}*/

void RNNBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in hx,
        _megdnn_tensor_in dy, _megdnn_tensor_in dhy, _megdnn_tensor_in flatten_weights,
        _megdnn_tensor_in reserve_space, _megdnn_tensor_out dx, _megdnn_tensor_out dhx,
        _megdnn_tensor_out dw, _megdnn_workspace workspace) {
    Handle* handle = this->handle();
    size_t seq_len = x.layout.shape[0];
    auto desc_holder = rnn::get_RNNDescHolder_v6(handle, param(), x.layout);
    auto x_desc_arr_ptr = rnn::get_descs(desc_holder.x_descs).data();
    auto y_desc_arr_ptr = rnn::get_descs(desc_holder.y_descs).data();
    RNNWeightFilterDesc w_desc;
    w_desc.set(flatten_weights.layout);

    cudnn_check(cudnnRNNBackwardData(
            cudnn_handle(handle), desc_holder.rnn_desc.desc, seq_len, y_desc_arr_ptr,
            y.raw_ptr(), y_desc_arr_ptr, dy.raw_ptr(), desc_holder.hy_desc.desc,
            dhy.raw_ptr(), desc_holder.cy_desc.desc, NULL, w_desc.desc,
            flatten_weights.raw_ptr(), desc_holder.hx_desc.desc, hx.raw_ptr(),
            desc_holder.cx_desc.desc, NULL, x_desc_arr_ptr, dx.raw_ptr(),
            desc_holder.hx_desc.desc, dhx.raw_ptr(), desc_holder.cx_desc.desc, NULL,
            workspace.raw_ptr, desc_holder.workspace_size, reserve_space.raw_ptr(),
            desc_holder.reserveSpace_size));

    cudnn_check(cudnnRNNBackwardWeights(
            cudnn_handle(handle), desc_holder.rnn_desc.desc, seq_len, x_desc_arr_ptr,
            x.raw_ptr(), desc_holder.hx_desc.desc, hx.raw_ptr(), y_desc_arr_ptr,
            y.raw_ptr(), workspace.raw_ptr, desc_holder.workspace_size, w_desc.desc,
            dw.raw_ptr(), reserve_space.raw_ptr(), desc_holder.reserveSpace_size));
}

size_t RNNBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
        const TensorLayout& dy, const TensorLayout& dhy,
        const TensorLayout& flatten_weights, const TensorLayout& reserve_space,
        const TensorLayout& dx, const TensorLayout& dhx, const TensorLayout& dw) {
    auto desc_holder = rnn::get_RNNDescHolder_v6(this->handle(), param(), x);
    return desc_holder.workspace_size;
}

}  // namespace cuda
}  // namespace megdnn
