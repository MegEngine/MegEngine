/**
 * \file dnn/src/cuda/rnn/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {
namespace rnn {
// v8, not for now
/*struct RNNForwardDescHolder {

            int32_t* devSeqLengths;
            cudnnRNNMode_t mode;
            cudnnForwardMode_t fwdMode;
            RNNDesc rnn_desc;
            DropoutDesc dropout_desc;
            RNNDataDesc x_desc, y_desc;
            TensorDesc h_desc;
            size_t weight_size, workspace_size, reserveSpace_size;

    RNNForwardDescHolder(Handle* handle, size_t seq_len, size_t batch_size, size_t
hidden_size, size_t input_size, size_t proj_size, size_t num_layers, bool bidirectional,
                                                     bool bias, DType dtype,
cudnnRNNMode_t _mode, cudnnForwardMode_t _fwdMode); ~RNNForwardDescHolder();
};*/

struct RNNForwardDescHolder_v6 {
    cudnnRNNMode_t mode;
    RNNDesc rnn_desc;
    int seq_len;
    DropoutDesc dropout_desc;
    std::vector<TensorDesc> x_descs, y_descs;
    TensorDesc hx_desc, cx_desc, hy_desc, cy_desc;

    size_t workspace_size, reserveSpace_size;

    RNNForwardDescHolder_v6(
            Handle* handle, size_t seq_len, size_t batch_size, size_t hidden_size,
            size_t input_size, size_t proj_size, size_t num_layers, bool bidirectional,
            bool bias, DType dtype, cudnnRNNMode_t _mode);
};

RNNForwardDescHolder_v6 get_RNNDescHolder_v6(
        Handle* handle, megdnn::RNNForward::Param& _param, const TensorLayout& input);
std::vector<cudnnTensorDescriptor_t> get_descs(const std::vector<TensorDesc>& descs);
}  // namespace rnn
}  // namespace cuda
}  // namespace megdnn