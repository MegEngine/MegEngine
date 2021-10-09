/**
 * \file dnn/src/naive/rnn/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/rnn/opr_impl.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs/base.h"
#include "megdnn/oprs/general.h"
#include "src/common/opr_delegate.h"
#include "src/common/rnn.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/matrix_mul/opr_impl.h"
#include "src/naive/rnn/funcs.h"
#include "src/naive/rnn/rnn.h"

#include <cstring>

namespace megdnn {
namespace naive {

using rnn::RNNCellWeightWrapper;

void RNNImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in hx,
        _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
        _megdnn_tensor_out hy, _megdnn_tensor_out reserve_space,
        _megdnn_workspace workspace) {
    auto _param = param();
    size_t D = _param.bidirectional ? 2 : 1;
    size_t num_layers = _param.num_layers;
    size_t input_size = input.layout.shape[2];
    std::vector<RNNCellWeightWrapper> cells;
    size_t used_workspace_size = rnn::get_cells<RNNCellWeightWrapper>(
            D, num_layers, input_size, _param.hidden_size, _param.bias, cells,
            flatten_weights, workspace);

    Workspace new_workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    TensorNDArray states, states_new;
    states.push_back(hx);
    states_new.push_back(hy);
    rnn::exec_internal<RNNCellWeightWrapper, RNNCellForward>(
            cells, input, states, states_new, output, reserve_space, num_layers, D,
            this->handle(), new_workspace);
}

size_t RNNImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& hx,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout& hy, const TensorLayout& reserve_space) {
    size_t workspace_size = rnn::get_workspace_in_bytes<RNNCellForward>(
            input, flatten_weights, param().hidden_size, param().bidirectional ? 2 : 1,
            this->handle());
    if (!param().bias) {  // use fake bias (all 0)
        TensorLayout bias_layout = {{param().hidden_size}, flatten_weights.dtype};
        workspace_size += bias_layout.span().dist_byte();
    }
    workspace_size += output.span().dist_byte();
    return workspace_size;
}

size_t RNNImpl::get_reserve_size_in_bytes(const TensorLayout& input) {
    size_t num_layers = param().num_layers;
    size_t D = param().bidirectional ? 2 : 1;
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    TensorLayout state_layout{{batch_size, param().hidden_size}, input.dtype};
    return num_layers * D * seq_len * state_layout.span().dist_byte();
}

void RNNBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in hx,
        _megdnn_tensor_in dy, _megdnn_tensor_in dhy, _megdnn_tensor_in flatten_weights,
        _megdnn_tensor_in reserve_space, _megdnn_tensor_out dx, _megdnn_tensor_out dhx,
        _megdnn_tensor_out dw, _megdnn_workspace workspace) {
    TensorNDArray layer_inputs;
    // layer_inputs.push_back(x);
    TensorNDArray layer_outputs;
    std::vector<std::vector<TensorNDArray>> cell_seq_states;
    size_t num_layers = param().num_layers;
    size_t D = param().bidirectional ? 2 : 1;
    // size_t seq_len = x.layout.shape[0];
    // size_t batch_size = x.layout.shape[1];
    size_t input_size = x.layout.shape[2];
    size_t hidden_size = param().hidden_size;
    size_t used_workspace_size = 0;

    // get cells
    std::vector<RNNCellWeightWrapper> cells;
    // workspace_ptr = static_cast<uint8_t*>(workspace_ptr) +
    used_workspace_size += rnn::get_cells(
            D, num_layers, input_size, hidden_size, param().bias, cells,
            flatten_weights, workspace);

    // extract intermedia states from reserve space
    /*for (int layer = 0; layer < num_layers; ++layer) {
            TensorND layer_output{workspace_ptr, y.layout};
            workspace_ptr = static_cast<uint8_t*>(workspace_ptr) +
    layer_output.layout.span().dist_byte(); for (int d = 0; d < D; ++d) {
                    cell_seq_states.push_back(std::vector<TensorNDArray>());
                    // reverse direction is stored with reversed order of sequence order
                    for (int i = 0; i < seq_len; ++i) {
                            size_t step = i;
                            if (d == 1) step = seq_len - i - 1;
                            size_t offset = ((layer * D + d) * seq_len + step) *
    cell_output_layout.span().dist_byte(); TensorND
    hy{static_cast<uint8_t*>(reserve_space.raw_ptr) + offset, cell_output_layout};
                            // states
                            cell_seq_states[cell_seq_states.size() - 1].push_back({hy});
                            // output
                            offset = i * D * cell_output_layout.span().dist_byte();
                            memcpy(static_cast<uint8_t*>(layer_output.raw_ptr) + offset,
                                       hy.raw_ptr, hy.layout.span().dist_byte());
                    }
            }
            cell_seq_outputs.push_back(layer_output);
            if (layer != num_layers - 1) layer_inputs.push_back(layer_output);
    }*/
    // nonlinear mode
    param::RNNCell::NonlineMode nonlineMode;
    using ModeRNN = param::RNN::NonlineMode;
    using ModeRNNCell = param::RNNCell::NonlineMode;
    switch (param().nonlineMode) {
        case ModeRNN::RELU:
            nonlineMode = ModeRNNCell::RELU;
            break;
        case ModeRNN::TANH:
            nonlineMode = ModeRNNCell::TANH;
            break;
    }

    // get formatted inputs
    Workspace new_workspace = Workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    used_workspace_size += rnn::get_inputs_for_exec<RNNCellWeightWrapper>(
            x, y, reserve_space, num_layers, D, hidden_size, cells, layer_inputs,
            layer_outputs, cell_seq_states, nonlineMode, new_workspace);

    // dhy arr, dhx arr
    TensorNDArray dhy_arr = {dhy}, dhx_arr = {dhx};

    // exec
    /*size_t used_workspace_size = static_cast<uint8_t*>(workspace_ptr) -
            static_cast<uint8_t*>((void*)workspace.raw_ptr);*/
    new_workspace = Workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    rnn::backward_exec_internal<RNNCellWeightWrapper>(
            cells, D, num_layers, input_size, param().bias, nonlineMode, layer_inputs,
            layer_outputs, cell_seq_states, dy, dhy_arr, dx, dhx_arr, dw,
            this->handle(), new_workspace);
}

size_t RNNBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
        const TensorLayout& dy, const TensorLayout& dhy,
        const TensorLayout& flatten_weights, const TensorLayout& reserve_space,
        const TensorLayout& dx, const TensorLayout& dhx, const TensorLayout& dw) {
    size_t D = param().bidirectional ? 2 : 1;
    size_t num_layers = param().num_layers;
    size_t hidden_size = param().hidden_size;
    size_t gate_hidden_size = hidden_size;
    size_t max_input_size = std::max(x.shape[2], D * hidden_size);

    size_t workspace_size = RNNCellWeightWrapper::backward_workspace_size_in_bytes(
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
