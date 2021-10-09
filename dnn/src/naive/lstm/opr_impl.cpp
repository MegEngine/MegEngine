/**
 * \file dnn/src/naive/lstm/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/lstm/opr_impl.h"
#include "src/naive/rnn/funcs.h"
#include "src/naive/rnn/rnn.h"

namespace megdnn {
namespace naive {
using rnn::LSTMCellWeightWrapper;

void LSTMImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in hx, _megdnn_tensor_in cx,
        _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
        _megdnn_tensor_out hy, _megdnn_tensor_out cy, _megdnn_tensor_out reserve_space,
        _megdnn_workspace workspace) {
    auto _param = param();
    size_t D = _param.bidirectional ? 2 : 1;
    size_t num_layers = _param.num_layers;
    size_t input_size = input.layout.shape[2];
    std::vector<LSTMCellWeightWrapper> cells;
    size_t used_workspace_size = rnn::get_cells<LSTMCellWeightWrapper>(
            D, num_layers, input_size, _param.hidden_size, _param.bias, cells,
            flatten_weights, workspace);

    Workspace new_workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    TensorNDArray states = {hx, cx}, states_new = {hy, cy};
    rnn::exec_internal<LSTMCellWeightWrapper, LSTMCellForward>(
            cells, input, states, states_new, output, reserve_space, num_layers, D,
            this->handle(), new_workspace);
}

size_t LSTMImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout& hy, const TensorLayout& cy,
        const TensorLayout& reserve_space) {
    size_t workspace_size = rnn::get_workspace_in_bytes<LSTMCellForward>(
            input, flatten_weights, param().hidden_size, param().bidirectional ? 2 : 1,
            this->handle());
    if (!param().bias) {  // use fake bias (all 0)
        TensorLayout bias_layout = {{param().hidden_size * 4}, flatten_weights.dtype};
        workspace_size += bias_layout.span().dist_byte();
    }
    workspace_size += output.span().dist_byte();
    return workspace_size;
}

size_t LSTMImpl::get_reserve_size_in_bytes(const TensorLayout& input) {
    size_t num_layers = param().num_layers;
    size_t D = param().bidirectional ? 2 : 1;
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    TensorLayout state_layout{{batch_size, param().hidden_size}, input.dtype};
    // 2 for hidden state and cell state
    return 2 * num_layers * D * seq_len * state_layout.span().dist_byte();
}

void LSTMBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in hx,
        _megdnn_tensor_in cx, _megdnn_tensor_in dy, _megdnn_tensor_in dhy,
        _megdnn_tensor_in dcy, _megdnn_tensor_in flatten_weights,
        _megdnn_tensor_in reserve_space, _megdnn_tensor_out dx, _megdnn_tensor_out dhx,
        _megdnn_tensor_out dcx, _megdnn_tensor_out dw, _megdnn_workspace workspace) {
    TensorNDArray layer_inputs;
    TensorNDArray layer_outputs;
    std::vector<std::vector<TensorNDArray>> cell_seq_states;
    size_t num_layers = param().num_layers;
    size_t D = param().bidirectional ? 2 : 1;
    size_t input_size = x.layout.shape[2];
    size_t hidden_size = param().hidden_size;
    size_t used_workspace_size = 0;

    // get cells
    std::vector<LSTMCellWeightWrapper> cells;
    used_workspace_size += rnn::get_cells<LSTMCellWeightWrapper>(
            D, num_layers, input_size, hidden_size, param().bias, cells,
            flatten_weights, workspace);

    // get formatted inputs
    Workspace new_workspace = Workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    used_workspace_size += rnn::get_inputs_for_exec<LSTMCellWeightWrapper>(
            x, y, reserve_space, num_layers, D, hidden_size, cells, layer_inputs,
            layer_outputs, cell_seq_states, param::RNNCell::NonlineMode::IDENTITY,
            new_workspace);

    // dhy arr, dhx arr
    TensorNDArray dhy_arr = {dhy, dcy}, dhx_arr = {dhx, dcx};

    // exec
    new_workspace = Workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    rnn::backward_exec_internal<LSTMCellWeightWrapper>(
            cells, D, num_layers, input_size, param().bias,
            param::RNNCell::NonlineMode::IDENTITY, layer_inputs, layer_outputs,
            cell_seq_states, dy, dhy_arr, dx, dhx_arr, dw, this->handle(),
            new_workspace);
}

size_t LSTMBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
        const TensorLayout& cx, const TensorLayout& dy, const TensorLayout& dhy,
        const TensorLayout& dcy, const TensorLayout& flatten_weights,
        const TensorLayout& reserve_space, const TensorLayout& dx,
        const TensorLayout& dhx, const TensorLayout& dcx, const TensorLayout& dw) {
    size_t D = param().bidirectional ? 2 : 1;
    size_t num_layers = param().num_layers;
    size_t hidden_size = param().hidden_size;
    size_t gate_hidden_size = hidden_size * 4;
    size_t max_input_size = std::max(x.shape[2], D * hidden_size);

    size_t workspace_size = LSTMCellWeightWrapper::backward_workspace_size_in_bytes(
            this->handle(), x.shape[1], param().hidden_size, max_input_size, x.dtype);
    if (!param().bias) {  // use fake bias (all 0)
        TensorLayout bias_layout = {{gate_hidden_size}, flatten_weights.dtype};
        workspace_size += bias_layout.span().dist_byte() *
                          2;  // times 2 because another bias is allocated in
                              // backward_exec_internal
    }
    workspace_size += num_layers * y.span().dist_byte();
    // add back exec workspace size
    workspace_size += y.span().dist_byte() * 2;
    workspace_size += x.span().dist_byte() * 2;
    TensorLayout wih{{gate_hidden_size, max_input_size}, flatten_weights.dtype};
    TensorLayout whh{{gate_hidden_size, hidden_size}, flatten_weights.dtype};
    TensorLayout bias{{gate_hidden_size}, flatten_weights.dtype};
    workspace_size += wih.span().dist_byte();
    workspace_size += whh.span().dist_byte();
    workspace_size += bias.span().dist_byte();
    return workspace_size;
}
}  // namespace naive

}  // namespace megdnn