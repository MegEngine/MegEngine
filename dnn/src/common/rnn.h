/**
 * \file dnn/src/common/rnn.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/base.h"
#include "megdnn/oprs/general.h"

namespace megdnn {
namespace rnn {
using Param = param::RNN;

size_t get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayoutArray& weight_ih,
        const TensorLayoutArray& states, const TensorLayoutArray& weight_hh,
        const TensorLayoutArray& bias, const TensorLayout& output,
        const TensorLayoutArray& states_new, const Param& param, Handle* handle);

void exec(
        _megdnn_tensor_in input, _megdnn_in const TensorNDArray& weight_ih,
        _megdnn_in const TensorNDArray& states,
        _megdnn_in const TensorNDArray& weight_hh, _megdnn_in const TensorNDArray& bias,
        _megdnn_tensor_out output, _megdnn_out const TensorNDArray& states_new,
        _megdnn_workspace workspace, const Param& param, Handle* handle);
}  // namespace rnn
}  // namespace megdnn