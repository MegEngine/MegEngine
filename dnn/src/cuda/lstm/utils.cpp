/**
 * \file dnn/src/cuda/lstm/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/lstm/utils.h"
#include "src/cuda/utils.h"

#include <cudnn.h>

namespace megdnn {
namespace cuda {
namespace lstm {

RNNForwardDescHolder_v6 get_RNNDescHolder_v6(
        Handle* handle, megdnn::LSTMForward::Param& _param, const TensorLayout& input) {
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    size_t input_size = input.shape[2];

    cudnnRNNMode_t mode = CUDNN_LSTM;

    using FwdMode = param::LSTM::FwdMode;

    RNNForwardDescHolder_v6 desc_holder(
            handle, seq_len, batch_size, _param.hidden_size, input_size,
            _param.proj_size, _param.num_layers, _param.bidirectional, _param.bias,
            input.dtype, mode);
    return desc_holder;
}

}  // namespace lstm
}  // namespace cuda
}  // namespace megdnn