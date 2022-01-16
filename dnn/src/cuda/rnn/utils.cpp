/**
 * \file dnn/src/cuda/rnn/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/rnn/utils.h"
#include "src/cuda/utils.h"

#include <cudnn.h>

namespace megdnn {
namespace cuda {
namespace rnn {
/*RNNForwardDescHolder::RNNForwardDescHolder(Handle* handle, size_t seq_len, size_t
batch_size, size_t hidden_size, size_t input_size, size_t proj_size, size_t num_layers,
bool bidirectional, bool bias, DType dtype, cudnnRNNMode_t _mode, cudnnForwardMode_t
_fwdMode) : mode(_mode), fwdMode(_fwdMode)
{
        size_t D = bidirectional ? 2 : 1;

        // TODO: set dropout to 0 in inference mode
        dropout_desc.set_no_dropout(handle);

        // seq len is unified (not packed)
        // cuda_check(cudaMalloc((void**)&devSeqLengths, sizeof(int32_t) * batch_size));
        devSeqLengths = (int32_t*)malloc(sizeof(int32_t) * batch_size);
        for (size_t i = 0; i < batch_size; ++i) devSeqLengths[i] = seq_len;

        // proj size should be smaller than hidden size according to cudnn api
        // otherwise it is disabled
        proj_size = (proj_size > hidden_size || proj_size == 0) ? hidden_size :
proj_size; rnn_desc.set( input_size, hidden_size, proj_size, num_layers, bidirectional,
bias, dtype, mode, dropout_desc, handle
        );

        x_desc.set(batch_size, input_size, seq_len, devSeqLengths, dtype);
        y_desc.set(batch_size, D * proj_size, seq_len,
                           devSeqLengths, dtype);
        h_desc.set_nd(TensorLayout(TensorShape{D * num_layers, batch_size, proj_size},
dtype));

        cudnn_check(cudnnGetRNNWeightSpaceSize(cudnn_handle(handle), rnn_desc.desc,
&weight_size));

        cudnn_check(cudnnGetRNNTempSpaceSizes(
                cudnn_handle(handle), rnn_desc.desc, fwdMode, x_desc.desc,
&workspace_size, &reserveSpace_size
        ));
}

RNNForwardDescHolder::~RNNForwardDescHolder() {
        // cuda_check(cudaFree(devSeqLengths));
        free(devSeqLengths);
}*/

RNNForwardDescHolder_v6::RNNForwardDescHolder_v6(
        Handle* handle, size_t seq_len, size_t batch_size, size_t hidden_size,
        size_t input_size, size_t proj_size, size_t num_layers, bool bidirectional,
        bool bias, DType dtype, cudnnRNNMode_t _mode)
        : mode(_mode), seq_len(seq_len) {
    size_t D = bidirectional ? 2 : 1;

    // TODO: set dropout to 0 in inference mode
    dropout_desc.set_no_dropout(handle);

    proj_size = (proj_size > hidden_size || proj_size == 0) ? hidden_size : proj_size;
    rnn_desc.set(
            input_size, hidden_size, proj_size, num_layers, bidirectional, bias, dtype,
            mode, dropout_desc, handle);

    x_descs.resize(seq_len);
    y_descs.resize(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        x_descs[i].set_nd(TensorLayout(TensorShape{batch_size, input_size}, dtype), 3);
        y_descs[i].set_nd(
                TensorLayout(TensorShape{batch_size, D * hidden_size}, dtype), 3);
    }

#define SET_H(_var)           \
    _var.set_nd(TensorLayout( \
            TensorShape{D * num_layers, batch_size, hidden_size}, dtype));

    SET_H(hx_desc)
    SET_H(cx_desc)
    SET_H(hy_desc)
    SET_H(cy_desc)
#undef SET_H

    std::vector<cudnnTensorDescriptor_t> x_desc_arr = get_descs(x_descs);
    cudnn_check(cudnnGetRNNWorkspaceSize(
            cudnn_handle(handle), rnn_desc.desc, seq_len, x_desc_arr.data(),
            &workspace_size));

    cudnn_check(cudnnGetRNNTrainingReserveSize(
            cudnn_handle(handle), rnn_desc.desc, seq_len, x_desc_arr.data(),
            &reserveSpace_size));
}

RNNForwardDescHolder_v6 get_RNNDescHolder_v6(
        Handle* handle, megdnn::RNNForward::Param& _param, const TensorLayout& input) {
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    size_t input_size = input.shape[2];

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

    RNNForwardDescHolder_v6 desc_holder(
            handle, seq_len, batch_size, _param.hidden_size, input_size,
            _param.proj_size, _param.num_layers, _param.bidirectional, _param.bias,
            input.dtype, mode);
    return desc_holder;
}

std::vector<cudnnTensorDescriptor_t> get_descs(const std::vector<TensorDesc>& descs) {
    std::vector<cudnnTensorDescriptor_t> r;
    r.reserve(descs.size());
    for (auto& desc : descs) {
        r.emplace_back(desc.desc);
    }
    return r;
}
}  // namespace rnn
}  // namespace cuda
}  // namespace megdnn