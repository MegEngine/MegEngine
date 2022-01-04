/**
 * \file dnn/src/arm_common/lstm/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/lstm/opr_impl.h"
#include "./lstm_utils.h"
#include "src/arm_common/lstm_cell/opr_impl.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_lstm)

using namespace megdnn;
using namespace arm_common;

void LSTMImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in hx, _megdnn_tensor_in cx,
        _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
        _megdnn_tensor_out hy, _megdnn_tensor_out cy, _megdnn_tensor_out,
        _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_arm_common_lstm, midout_iv(0)) {
        size_t dir_size = param().bidirectional ? 2 : 1;
        size_t num_layers = param().num_layers;
        size_t hidden_size = param().hidden_size;

        size_t seq_len = input.layout.shape[0];
        size_t batch_size = input.layout.shape[1];
        size_t input_size = input.layout.shape[2];

        //! in order to support input ptr change in record, so this task should be
        //! dispatch to device
        auto&& cell_weights = get_all_cells<LstmCellWeight>(
                dir_size, num_layers, input_size, hidden_size, param().bias,
                flatten_weights);
        auto&& cell_states_in = get_all_status<LstmStates>(
                hx, cx, hidden_size, batch_size, num_layers, dir_size, hx.layout.dtype);
        auto&& cell_states_out = get_all_status<LstmStates>(
                hy, cy, hidden_size, batch_size, num_layers, dir_size, hy.layout.dtype);
        auto&& inputs = split_tensor(
                input, seq_len,
                TensorLayout{{batch_size, input_size}, input.layout.dtype});
        auto&& outputs = split_tensor(
                output, seq_len,
                TensorLayout{
                        {batch_size, dir_size * hidden_size}, output.layout.dtype});

        auto workspace_bundle = get_workspace_bundle<LSTMCell>(
                input.layout, output.layout, flatten_weights.layout, hidden_size,
                dir_size, LstmStates::nr_states());

        workspace_bundle.set(workspace.raw_ptr);
        exec_kernel<LstmCellWeight, LSTMCell, LstmStates>(
                cell_weights, inputs, cell_states_in, cell_states_out, outputs,
                num_layers, dir_size, handle(), workspace_bundle);
    }
    MIDOUT_END();
}

size_t LSTMImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout&, const TensorLayout&,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout&, const TensorLayout&, const TensorLayout&) {
    MIDOUT_BEGIN(megdnn_arm_common_lstm, midout_iv(1)) {
        size_t dir_size = param().bidirectional ? 2 : 1;
        size_t hidden_size = param().hidden_size;

        auto bundle = get_workspace_bundle<LSTMCell>(
                input, output, flatten_weights, hidden_size, dir_size,
                LstmStates::nr_states());
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
}

// vim: syntax=cpp.doxygen
