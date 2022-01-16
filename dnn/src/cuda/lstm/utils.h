/**
 * \file dnn/src/cuda/lstm/utils.h
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
#include "src/cuda/rnn/utils.h"

namespace megdnn {
namespace cuda {
namespace lstm {
using megdnn::cuda::rnn::RNNForwardDescHolder_v6;
RNNForwardDescHolder_v6 get_RNNDescHolder_v6(
        Handle* handle, megdnn::LSTMForward::Param& _param, const TensorLayout& input);
}  // namespace lstm
}  // namespace cuda
}  // namespace megdnn