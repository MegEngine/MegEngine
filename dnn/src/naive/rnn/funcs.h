/**
 * \file dnn/src/naive/rnn/funcs.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef _RNN_H
#define _RNN_H
#include "megdnn/oprs.h"
namespace megdnn {
namespace naive {
namespace rnn {

template <typename CellOpr>
void cell_opr_exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
        _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in bias_hh, const TensorNDArray& states,
        TensorNDArray& states_new, _megdnn_workspace workspace, Handle* handle);

template <typename CellOpr>
size_t cell_opr_get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& weight_hh, const TensorLayout& bias_ih,
        const TensorLayout& bias_hh, const TensorLayout& hx, Handle* handle);

template <typename CellOpr>
size_t get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& flatten_weights,
        size_t hidden_size,
        size_t D,  // num_directions
        Handle* handle);

template <class Cell, typename CellOpr>
void exec_internal(
        std::vector<Cell>& cells, _megdnn_tensor_in input, const TensorNDArray& states,
        TensorNDArray& states_new, _megdnn_tensor_out output,
        _megdnn_tensor_out reserve_space, size_t num_layers,
        size_t D,  // D is num_directions
        Handle* handle, _megdnn_workspace workspace);

template <class Cell>
size_t get_cells(
        size_t D, size_t num_layers, size_t input_size, size_t hidden_size, bool bias,
        std::vector<Cell>& cells, _megdnn_tensor_in flatten_weights,
        _megdnn_workspace workspace);

template <class Cell>
size_t get_inputs_for_exec(
        _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in reserve_space,
        size_t num_layers, size_t D, size_t hidden_size, const std::vector<Cell>& cells,
        TensorNDArray& layer_inputs, TensorNDArray& layer_outputs,
        std::vector<std::vector<TensorNDArray>>& cell_seq_states,
        param::RNNCell::NonlineMode nonlineMode, _megdnn_workspace workspace);

template <class Cell>
void backward_exec_internal(
        std::vector<Cell>& cells, size_t D, size_t num_layers, size_t input_size,
        bool bias, param::RNNCell::NonlineMode nonlineMode,
        const TensorNDArray& layer_inputs, const TensorNDArray& layer_outputs,
        const std::vector<std::vector<TensorNDArray>>& cell_seq_states,
        _megdnn_tensor_in dy, const TensorNDArray& dhy, _megdnn_tensor_out dx,
        TensorNDArray& dstates, _megdnn_tensor_out dw, Handle* handle,
        _megdnn_workspace workspace);

}  // namespace rnn
}  // namespace naive
}  // namespace megdnn

#include "funcs.tpp"
#endif
