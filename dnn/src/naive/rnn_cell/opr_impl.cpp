/**
 * \file dnn/src/naive/rnn_cell/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/rnn_cell/opr_impl.h"
#include "src/common/rnn_cell.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_rnncell_fwd)

namespace megdnn {
namespace naive {
size_t RNNCellImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& dst) {
    return megdnn::rnn_cell::get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, dst, handle());
}

void RNNCellImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_rnncell_fwd) {
        megdnn::rnn_cell::exec(
                input, weight_ih, bias_ih, hx, weight_hh, bias_hh, dst, workspace,
                param().nonlineMode, handle());
    }
    MIDOUT_END();
}
}  // namespace naive
}  // namespace megdnn