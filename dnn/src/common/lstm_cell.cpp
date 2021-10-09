/**
 * \file dnn/src/common/lstm_cell.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/lstm_cell.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void LSTMCell::deduce_layout(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& cx, TensorLayout& h_new, TensorLayout& c_new,
        TensorLayout& gates) {
    // size_t batch_size = hx.shape[0];
    // size_t hidden_size = hx.shape[1];
    h_new = TensorLayout(hx, hx.dtype);
    c_new = TensorLayout(cx, cx.dtype);
    auto opr = handle()->create_operator<RNNCellForward>();
    opr->param().nonlineMode = param::RNNCell::NonlineMode::IDENTITY;
    opr->deduce_layout(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, gates);
}

void LSTMCell::check_exec(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& cx, const TensorLayout& h_new, const TensorLayout& c_new,
        const TensorLayout& gates, size_t workspace_in_bytes) {
    TensorLayout h_new_expected, c_new_expected, gates_expected;
    deduce_layout(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new_expected,
            c_new_expected, gates_expected);
    megdnn_assert_eq_layout(h_new_expected, h_new);
    megdnn_assert_eq_layout(c_new_expected, c_new);
    megdnn_assert_eq_layout(gates_expected, gates);

    auto required_workspace_in_bytes = get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new, gates);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

namespace megdnn {
namespace lstm_cell {

size_t get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& cx, const TensorLayout& h_new, const TensorLayout& c_new,
        const TensorLayout& gates, Handle* handle) {
    TensorLayout tmp_layout;
    auto opr = handle->create_operator<RNNCellForward>();
    opr->param().nonlineMode = param::RNNCell::NonlineMode::IDENTITY;
    opr->deduce_layout(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, tmp_layout);
    size_t rnn_cell_need = opr->get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, gates);
    size_t lstm_cell_need = tmp_layout.span().dist_byte();
    return rnn_cell_need > lstm_cell_need ? rnn_cell_need : lstm_cell_need;
}

void exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_in cx, _megdnn_tensor_out h_new, _megdnn_tensor_out c_new,
        _megdnn_tensor_out gates, _megdnn_workspace workspace, Handle* handle) {
    auto opr = handle->create_operator<RNNCellForward>();
    opr->param().nonlineMode = param::RNNCell::NonlineMode::IDENTITY;
    /*TensorLayout tmp_layout;
    opr->deduce_layout(input.layout, weight_ih.layout,
                                       hx.layout, weight_hh.layout,
                                       bias.layout, tmp_layout);
    auto workspace_ptr = workspace.raw_ptr;
    // TensorND tmp{static_cast<void*>(workspace.raw_ptr), tmp_layout};
    TensorND tmp{workspace_ptr, tmp_layout};
    auto new_workspace = Workspace{workspace_ptr + tmp.layout.span().dist_byte(),
                                                               workspace.size -
    tmp.layout.span().dist_byte()};*/
    // opr->exec(input, weight_ih, hx, weight_hh, bias, tmp, new_workspace);
    opr->exec(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, gates, workspace);
    // activation
    // size_t batch_size = tmp.layout.shape[0];
    size_t batch_size = hx.layout.shape[0];
    size_t hidden_size = hx.layout.shape[1];
    // sigmoid: i f o
    // TensorLayout gates_ifo_layout{TensorShape({batch_size, hidden_size * 3}),
    // tmp.layout.dtype};
    TensorND tmp{static_cast<void*>(workspace.raw_ptr), gates.layout};
    TensorLayout gates_ifo_layout{
            TensorShape({batch_size, hidden_size * 3}), gates.layout.dtype};
    TensorND gates_ifo_origin{gates.raw_ptr(), gates_ifo_layout};
    TensorND gates_ifo{tmp.raw_ptr(), gates_ifo_layout};
    auto sigmoid = handle->create_operator<ElemwiseForward>();
    sigmoid->param().mode = Elemwise::Param::Mode::SIGMOID;
    sigmoid->exec({gates_ifo_origin}, gates_ifo);
    // tanh: g
    TensorLayout g_layout{TensorShape({batch_size, hidden_size}), gates.layout.dtype};
    TensorND g_origin{
            static_cast<char*>(gates.raw_ptr()) + gates_ifo_layout.span().dist_byte(),
            g_layout};
    TensorND g{
            static_cast<char*>(tmp.raw_ptr()) + gates_ifo_layout.span().dist_byte(),
            g_layout};
    auto tanh = handle->create_operator<ElemwiseForward>();
    tanh->param().mode = Elemwise::Param::Mode::TANH;
    tanh->exec({g_origin}, g);
    // extract i f o
    TensorND i{static_cast<char*>(tmp.raw_ptr()), g_layout};
    TensorND f{
            static_cast<char*>(tmp.raw_ptr()) + g_layout.span().dist_byte(), g_layout};
    TensorND o{
            static_cast<char*>(tmp.raw_ptr()) + g_layout.span().dist_byte() * 2,
            g_layout};
    // calculate new cell state
    auto elewise_mul_add = handle->create_operator<ElemwiseForward>();
    elewise_mul_add->param().mode = Elemwise::Param::Mode::FUSE_MUL_ADD4;
    elewise_mul_add->exec({f, cx, i, g}, c_new);
    // calculate new hidden state
    tanh->exec({c_new}, h_new);
    auto elewise_mul = handle->create_operator<ElemwiseForward>();
    elewise_mul->param().mode = Elemwise::Param::Mode::MUL;
    elewise_mul->exec({o, h_new}, h_new);
}

}  // namespace lstm_cell
}  // namespace megdnn
