/**
 * \file dnn/src/naive/lstm/template_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/rnn/funcs.h"

namespace megdnn {
namespace naive {
namespace rnn {

template <>
void cell_opr_exec<LSTMCellForward>(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
        _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in bias_hh, const TensorNDArray& states,
        TensorNDArray& states_new, _megdnn_workspace workspace, Handle* handle) {
    auto opr = handle->create_operator<LSTMCellForward>();
    TensorLayout gates, h_new, c_new;
    opr->deduce_layout(
            input.layout, weight_ih.layout, bias_ih.layout, states[0].layout,
            weight_hh.layout, bias_hh.layout, states[1].layout, h_new, c_new, gates);
    TensorND gates_tensor{workspace.raw_ptr, gates};
    _megdnn_workspace new_workspace = {
            workspace.raw_ptr + gates.span().dist_byte(),
            workspace.size - gates.span().dist_byte()};
    opr->exec(
            input, weight_ih, bias_ih, states[0], weight_hh, bias_hh, states[1],
            states_new[0], states_new[1], gates_tensor, new_workspace);
}

template <>
size_t cell_opr_get_workspace_in_bytes<LSTMCellForward>(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& weight_hh, const TensorLayout& bias_ih,
        const TensorLayout& bias_hh, const TensorLayout& hx, Handle* handle) {
    TensorLayout cx = hx;
    TensorLayout h_new, c_new, gates;
    auto cell_opr = handle->create_operator<LSTMCellForward>();
    cell_opr->deduce_layout(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new, gates);
    return cell_opr->get_workspace_in_bytes(
                   input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new,
                   gates) +
           gates.span().dist_byte();
}

}  // namespace rnn
}  // namespace naive
}  // namespace megdnn