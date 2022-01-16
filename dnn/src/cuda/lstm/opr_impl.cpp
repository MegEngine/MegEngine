/**
 * \file dnn/src/cuda/lstm/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/lstm/opr_impl.h"
#include "src/cuda/lstm/utils.h"
#include "src/cuda/utils.h"

#include <cudnn.h>

namespace megdnn {
namespace cuda {

void LSTMImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in hx, _megdnn_tensor_in cx,
        _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
        _megdnn_tensor_out hy, _megdnn_tensor_out cy, _megdnn_tensor_out reserve_space,
        _megdnn_workspace workspace) {
    Handle* handle = this->handle();

    rnn::RNNForwardDescHolder_v6 desc_holder =
            lstm::get_RNNDescHolder_v6(this->handle(), param(), input.layout);
    auto x_desc_arr = rnn::get_descs(desc_holder.x_descs);
    auto y_desc_arr = rnn::get_descs(desc_holder.y_descs);
    RNNWeightFilterDesc w_desc;
    w_desc.set(flatten_weights.layout);

    if (param().fwd_mode == param::LSTM::FwdMode::TRAINING) {
        cudnn_check(cudnnRNNForwardTraining(
                cudnn_handle(handle), desc_holder.rnn_desc.desc, desc_holder.seq_len,
                x_desc_arr.data(), input.raw_ptr(), desc_holder.hx_desc.desc,
                hx.raw_ptr(), desc_holder.cx_desc.desc, cx.raw_ptr(), w_desc.desc,
                flatten_weights.raw_ptr(), y_desc_arr.data(), output.raw_ptr(),
                desc_holder.hy_desc.desc, hy.raw_ptr(), desc_holder.cy_desc.desc,
                cy.raw_ptr(), workspace.raw_ptr, desc_holder.workspace_size,
                reserve_space.raw_ptr(), desc_holder.reserveSpace_size));
    } else {
        cudnn_check(cudnnRNNForwardInference(
                cudnn_handle(handle), desc_holder.rnn_desc.desc, desc_holder.seq_len,
                x_desc_arr.data(), input.raw_ptr(), desc_holder.hx_desc.desc,
                hx.raw_ptr(), desc_holder.cx_desc.desc, nullptr, w_desc.desc,
                flatten_weights.raw_ptr(), y_desc_arr.data(), output.raw_ptr(),
                desc_holder.hy_desc.desc, hy.raw_ptr(), desc_holder.cy_desc.desc,
                cy.raw_ptr(), workspace.raw_ptr, desc_holder.workspace_size));
    }
}

size_t LSTMImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout& hy, const TensorLayout& cy,
        const TensorLayout& reserve_space) {
    rnn::RNNForwardDescHolder_v6 desc_holder =
            lstm::get_RNNDescHolder_v6(this->handle(), param(), input);
    return desc_holder.workspace_size;
}

size_t LSTMImpl::get_reserve_size_in_bytes(const TensorLayout& input) {
    rnn::RNNForwardDescHolder_v6 desc_holder =
            lstm::get_RNNDescHolder_v6(this->handle(), param(), input);
    return desc_holder.reserveSpace_size;
}

void LSTMBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in hx,
        _megdnn_tensor_in cx, _megdnn_tensor_in dy, _megdnn_tensor_in dhy,
        _megdnn_tensor_in dcy, _megdnn_tensor_in flatten_weights,
        _megdnn_tensor_in reserve_space, _megdnn_tensor_out dx, _megdnn_tensor_out dhx,
        _megdnn_tensor_out dcx, _megdnn_tensor_out dw, _megdnn_workspace workspace) {
    Handle* handle = this->handle();
    size_t seq_len = x.layout.shape[0];
    auto desc_holder = lstm::get_RNNDescHolder_v6(handle, param(), x.layout);
    auto x_desc_arr_ptr = rnn::get_descs(desc_holder.x_descs).data();
    auto y_desc_arr_ptr = rnn::get_descs(desc_holder.y_descs).data();
    RNNWeightFilterDesc w_desc;
    w_desc.set(flatten_weights.layout);

    cudnn_check(cudnnRNNBackwardData(
            cudnn_handle(handle), desc_holder.rnn_desc.desc, seq_len, y_desc_arr_ptr,
            y.raw_ptr(), y_desc_arr_ptr, dy.raw_ptr(), desc_holder.hy_desc.desc,
            dhy.raw_ptr(), desc_holder.cy_desc.desc, dcy.raw_ptr(), w_desc.desc,
            flatten_weights.raw_ptr(), desc_holder.hx_desc.desc, hx.raw_ptr(),
            desc_holder.cx_desc.desc, cx.raw_ptr(), x_desc_arr_ptr, dx.raw_ptr(),
            desc_holder.hx_desc.desc, dhx.raw_ptr(), desc_holder.cx_desc.desc,
            dcx.raw_ptr(), workspace.raw_ptr, desc_holder.workspace_size,
            reserve_space.raw_ptr(), desc_holder.reserveSpace_size));

    cudnn_check(cudnnRNNBackwardWeights(
            cudnn_handle(handle), desc_holder.rnn_desc.desc, seq_len, x_desc_arr_ptr,
            x.raw_ptr(), desc_holder.hx_desc.desc, hx.raw_ptr(), y_desc_arr_ptr,
            y.raw_ptr(), workspace.raw_ptr, desc_holder.workspace_size, w_desc.desc,
            dw.raw_ptr(), reserve_space.raw_ptr(), desc_holder.reserveSpace_size));
}

size_t LSTMBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
        const TensorLayout& cx, const TensorLayout& dy, const TensorLayout& dhy,
        const TensorLayout& dcy, const TensorLayout& flatten_weights,
        const TensorLayout& reserve_space, const TensorLayout& dx,
        const TensorLayout& dhx, const TensorLayout& dcx, const TensorLayout& dw) {
    auto desc_holder = lstm::get_RNNDescHolder_v6(this->handle(), param(), x);
    return desc_holder.workspace_size;
}

}  // namespace cuda
}  // namespace megdnn
