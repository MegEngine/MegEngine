/**
 * \file dnn/src/naive/rnn/funcs.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "funcs.h"

namespace megdnn {
namespace naive {
namespace rnn {

template <typename CellOpr>
size_t get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& flatten_weights,
        size_t hidden_size,
        size_t D,  // num_directions
        Handle* handle) {
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    size_t input_size = input.shape[2];
    size_t gate_hidden_size = flatten_weights.shape[0];
    // concat workspace
    TensorLayout direction_output_layout{
            TensorShape{seq_len, batch_size, hidden_size}, input.dtype};
    TensorLayout output_layout{{seq_len, batch_size, D * hidden_size}, input.dtype};
    TensorLayoutArray layer_layouts;
    for (size_t i = 0; i < D; ++i)
        layer_layouts.push_back(direction_output_layout);
    auto concat_opr = handle->create_operator<ConcatForward>();
    concat_opr->param().axis = -1;
    size_t concat_workspace =
            concat_opr->get_workspace_in_bytes(layer_layouts, output_layout);
    // cell workspace
    TensorLayout weight_ih{{gate_hidden_size, input_size}, flatten_weights.dtype};
    TensorLayout weight_hh{{gate_hidden_size, hidden_size}, flatten_weights.dtype};
    TensorLayout bias{{gate_hidden_size}, flatten_weights.dtype};
    TensorLayout hx{{batch_size, hidden_size}, input.dtype};
    size_t cell_workspace = cell_opr_get_workspace_in_bytes<CellOpr>(
            input, weight_ih, weight_hh, bias, bias, hx, handle);

    return std::max(cell_workspace, concat_workspace);
}

template <class Cell, typename CellOpr>
void exec_internal(
        std::vector<Cell>& cells, _megdnn_tensor_in input, const TensorNDArray& states,
        TensorNDArray& states_new, _megdnn_tensor_out output,
        _megdnn_tensor_out reserve_space, size_t num_layers,
        size_t D,  // D is num_directions
        Handle* handle, _megdnn_workspace workspace) {
    size_t seq_len = input.layout.shape[0];
    size_t batch_size = input.layout.shape[1];
    size_t input_size = input.layout.shape[2];
    size_t hidden_size = cells[0].weight_hh.layout.shape[1];
    TensorLayout cell_output_layout{
            TensorShape{batch_size, hidden_size}, states[0].layout.dtype};
    TensorLayout cell_first_input_layout{
            TensorShape{batch_size, input_size}, input.layout.dtype};
    TensorLayout cell_input_layout{
            TensorShape{batch_size, D * hidden_size}, input.layout.dtype};
    TensorLayout direction_output_layout{
            TensorShape{seq_len, batch_size, hidden_size}, output.layout.dtype};
    TensorND tmp_output{workspace.raw_ptr, output.layout};
    _megdnn_workspace new_workspace{
            workspace.raw_ptr + tmp_output.layout.span().dist_byte(),
            workspace.size - tmp_output.layout.span().dist_byte()};

    auto cell_opr = handle->create_operator<CellOpr>();
    auto copy_opr = handle->create_operator<TypeCvtForward>();

    // copy states to states_new
    for (size_t i = 0; i < states.size(); ++i)
        copy_opr->exec(states[i], states_new[i]);
    void* reserve_ptr = reserve_space.raw_ptr();

    // layer 1
    // TensorNDArray layer_outputs;
    for (size_t d = 0; d < D; ++d) {
        size_t cell_idx = d;
        auto& cell = cells[cell_idx];

        TensorNDArray cur_states;
        size_t states_offset = cell_idx * cell_output_layout.span().dist_byte();
        for (size_t i = 0; i < states.size(); ++i) {
            cur_states.push_back(TensorND{
                    static_cast<uint8_t*>(states_new[i].raw_ptr()) + states_offset,
                    cell_output_layout});
        }
        // TensorND direction_output_tensor{output.raw_ptr + d *
        // direction_output_layout.span().dist_byte(),
        // direction_output_layout};
        for (size_t i = 0; i < seq_len; ++i) {
            size_t step = d == 0 ? i : seq_len - 1 - i;
            TensorND step_input{
                    static_cast<uint8_t*>(input.raw_ptr()) +
                            step * cell_first_input_layout.span().dist_byte(),
                    cell_first_input_layout};
            TensorND step_output{
                    static_cast<uint8_t*>(output.raw_ptr()) +
                            (step * D + d) * cell_output_layout.span().dist_byte(),
                    cell_output_layout};
            // temporary states of each step (use reserve space)
            TensorNDArray tmp_states;
            for (size_t s = 0; s < cur_states.size(); ++s) {
                tmp_states.push_back(TensorND{reserve_ptr, cur_states[s].layout});
                size_t size_in_bytes = cur_states[s].layout.span().dist_byte();
                reserve_ptr = static_cast<uint8_t*>(reserve_ptr) + size_in_bytes;
            }
            cell_opr_exec<CellOpr>(
                    step_input, cell.weight_ih, cell.weight_hh, cell.bias_ih,
                    cell.bias_hh, cur_states, tmp_states, new_workspace, handle);
            // copy states to cur_states
            for (size_t s = 0; s < tmp_states.size(); ++s) {
                copy_opr->exec(tmp_states[s], cur_states[s]);
            }
            // copy h to step output
            copy_opr->exec(cur_states[0], step_output);
        }
    }

    for (size_t layer = 1; layer < num_layers; ++layer) {
        // TensorNDArray layer_outputs;

        for (size_t d = 0; d < D; ++d) {
            size_t cell_idx = layer * D + d;
            auto& cell = cells[cell_idx];

            TensorNDArray cur_states;
            size_t states_offset = cell_idx * cell_output_layout.span().dist_byte();
            for (size_t i = 0; i < states.size(); ++i) {
                cur_states.push_back(TensorND{
                        static_cast<uint8_t*>(states_new[i].raw_ptr()) + states_offset,
                        cell_output_layout});
            }
            // TensorND direction_output_tensor{output.raw_ptr + d *
            // direction_output_layout.span().dist_byte(),
            // direction_output_layout};

            for (size_t i = 0; i < seq_len; ++i) {
                size_t step = d == 0 ? i : seq_len - 1 - i;
                TensorND step_input{
                        static_cast<uint8_t*>(output.raw_ptr()) +
                                step * cell_input_layout.span().dist_byte(),
                        cell_input_layout};
                TensorND step_output{
                        static_cast<uint8_t*>(tmp_output.raw_ptr()) +
                                (step * D + d) * cell_output_layout.span().dist_byte(),
                        cell_output_layout};
                // temporary states of each step (use reserve space)
                TensorNDArray tmp_states;
                for (size_t s = 0; s < cur_states.size(); ++s) {
                    tmp_states.push_back(TensorND{reserve_ptr, cur_states[s].layout});
                    size_t size_in_bytes = cur_states[s].layout.span().dist_byte();
                    reserve_ptr = static_cast<uint8_t*>(reserve_ptr) + size_in_bytes;
                }
                cell_opr_exec<CellOpr>(
                        step_input, cell.weight_ih, cell.weight_hh, cell.bias_ih,
                        cell.bias_hh, cur_states, cur_states, new_workspace, handle);
                // copy states to cur_states
                for (size_t s = 0; s < tmp_states.size(); ++s) {
                    copy_opr->exec(tmp_states[s], cur_states[s]);
                }
                // copy h to step_output
                copy_opr->exec(cur_states[0], step_output);
            }
        }
        // copy layer output to output
        copy_opr->exec(tmp_output, output);
    }
    // output: [d0, d1, d0, d1 ...]
}

template <class Cell>
size_t get_cells(
        size_t D, size_t num_layers, size_t input_size, size_t hidden_size, bool bias,
        std::vector<Cell>& cells, _megdnn_tensor_in flatten_weights,
        _megdnn_workspace workspace) {
    cells.reserve(D * num_layers);
    void* weight_ptr = flatten_weights.raw_ptr();
    for (size_t layer = 0; layer < num_layers; ++layer) {
        for (size_t d = 0; d < D; ++d) {
            size_t cell_input_size = D * hidden_size;
            if (layer == 0)
                cell_input_size = input_size;
            Cell cell(
                    weight_ptr, hidden_size, cell_input_size, bias,
                    flatten_weights.layout.dtype, workspace);
            weight_ptr =
                    static_cast<uint8_t*>(weight_ptr) + cell.weight_size_in_bytes();
            cells.push_back(cell);
        }
    }
    // return used workspace
    return cells[0].workspace_size_in_bytes();
}

template <class Cell>
size_t get_inputs_for_exec(
        _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in reserve_space,
        size_t num_layers, size_t D, size_t hidden_size, const std::vector<Cell>& cells,
        TensorNDArray& layer_inputs, TensorNDArray& layer_outputs,
        std::vector<std::vector<TensorNDArray>>& cell_seq_states,
        param::RNNCell::NonlineMode nonlineMode, _megdnn_workspace workspace) {
    // return used workspace size

    layer_inputs.push_back(x);
    size_t seq_len = x.layout.shape[0];
    size_t batch_size = x.layout.shape[1];
    size_t num_states = cells[0].num_states();
    TensorLayout cell_output_layout{{batch_size, hidden_size}, y.layout.dtype};
    TensorLayout direction_output_layout{
            {seq_len, batch_size, hidden_size}, y.layout.dtype};
    void* workspace_ptr = workspace.raw_ptr;

    // extract intermedia states from reserve space
    for (int layer = 0; layer < num_layers; ++layer) {
        TensorND layer_output{workspace_ptr, y.layout};
        workspace_ptr = static_cast<uint8_t*>(workspace_ptr) +
                        layer_output.layout.span().dist_byte();
        for (int d = 0; d < D; ++d) {
            cell_seq_states.push_back(std::vector<TensorNDArray>());
            // reverse direction is stored with reversed order of sequence order
            for (int i = 0; i < seq_len; ++i) {
                size_t step = i;
                if (d == 1)
                    step = seq_len - i - 1;
                size_t offset = ((layer * D + d) * seq_len + step) *
                                cell_output_layout.span().dist_byte() * num_states;
                TensorNDArray cur_states;
                for (int s = 0; s < num_states; ++s) {
                    TensorND h{
                            static_cast<uint8_t*>(reserve_space.raw_ptr()) + offset +
                                    s * cell_output_layout.span().dist_byte(),
                            cell_output_layout};
                    cur_states.push_back(h);
                }
                TensorND hy{
                        static_cast<uint8_t*>(reserve_space.raw_ptr()) + offset,
                        cell_output_layout};  // the first hidden state is the output
                // states
                cell_seq_states[cell_seq_states.size() - 1].push_back(cur_states);
                // output
                offset = i * D * cell_output_layout.span().dist_byte();
                memcpy(static_cast<uint8_t*>(layer_output.raw_ptr()) + offset, hy.raw_ptr(),
                       hy.layout.span().dist_byte());
            }
        }
        layer_outputs.push_back(layer_output);
        if (layer != num_layers - 1)
            layer_inputs.push_back(layer_output);
    }
    return static_cast<uint8_t*>(workspace_ptr) -
           static_cast<uint8_t*>((void*)workspace.raw_ptr);
}

template <class Cell>
// using Cell = RNNCellWeightWrapper;
void backward_exec_internal(
        std::vector<Cell>& cells, size_t D, size_t num_layers, size_t input_size,
        bool bias, param::RNNCell::NonlineMode nonlineMode,
        const TensorNDArray& layer_inputs, const TensorNDArray& layer_outputs,
        const std::vector<std::vector<TensorNDArray>>& cell_seq_states,
        _megdnn_tensor_in dy, const TensorNDArray& dhy, _megdnn_tensor_out dx,
        TensorNDArray& dstates, _megdnn_tensor_out dw, Handle* handle,
        _megdnn_workspace workspace) {
    /*
        layer_inputs: array of input of each layer, element 0: [seq_len, batch_size,
       input_size], element others: [seq_len, batch_size, D * hidden_size]
       layer_outputs: array of outputs of each rnn. To access outputs of the cell at
       (layer, d), use layer_outputs[layer]. The shape is [seq_len, batch_size,
       output_size(D*hidden_size)] (in sequence order) cell_seq_states: arrray of states
       of each cell at each step. To access the states of the cell at (layer, d) at
       sequence step (step), use cell_seq_states[layer*D + d][step]
    */
    size_t seq_len = layer_inputs[0].layout.shape[0];
    size_t batch_size = layer_inputs[0].layout.shape[1];
    DType dtype = layer_inputs[0].layout.dtype;
    size_t cell_y_size =
            layer_outputs[0].layout.shape[2] / D;  // should all be the same
    size_t hidden_size = cell_y_size;
    TensorLayout cell_y_layout = {{batch_size, cell_y_size}, dtype};
    void* workspace_ptr = workspace.raw_ptr;

    TensorND layer_output_grad{
            workspace_ptr, {{seq_len, batch_size, D * hidden_size}, dtype}};
    workspace_ptr = static_cast<uint8_t*>(workspace_ptr) +
                    layer_output_grad.layout.span().dist_byte();
    memcpy(layer_output_grad.raw_ptr(), dy.raw_ptr(), dy.layout.span().dist_byte());
    TensorNDArray direction_dx_arr;  // for layer 1 to layer num_layers-1
    for (int i = 0; i < D; ++i) {
        TensorLayout direction_dx_layout{{seq_len, batch_size, hidden_size}, dtype};
        direction_dx_arr.push_back(TensorND(workspace_ptr, direction_dx_layout));
        workspace_ptr = static_cast<uint8_t*>(workspace_ptr) +
                        direction_dx_layout.span().dist_byte();
    }
    TensorNDArray L0_direction_dx_arr;
    for (int i = 0; i < D; ++i) {
        TensorLayout direction_dx_layout{{seq_len, batch_size, input_size}, dtype};
        L0_direction_dx_arr.push_back(TensorND(workspace_ptr, direction_dx_layout));
        workspace_ptr = static_cast<uint8_t*>(workspace_ptr) +
                        direction_dx_layout.span().dist_byte();
    }
    // cell states for each layer and each direction
    std::vector<TensorNDArray> dstates_arr;
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int d = 0; d < D; ++d) {
            TensorNDArray cell_states;
            cell_states.reserve(dstates.size());
            for (int i = 0; i < dstates.size(); ++i) {
                size_t offset = (layer * D + d) * cell_y_layout.span().dist_byte();
                TensorND dhx_cell{
                        static_cast<uint8_t*>(dstates[i].raw_ptr()) + offset,
                        cell_y_layout};
                memcpy(dhx_cell.raw_ptr(), static_cast<uint8_t*>(dhy[i].raw_ptr()) + offset,
                       cell_y_layout.span().dist_byte());
                cell_states.emplace_back(dhx_cell);
            }
            dstates_arr.push_back(cell_states);
        }
    }

    // init gradient on weight to zero
    memset(dw.raw_ptr(), 0, dw.layout.span().dist_byte());
    // use cells to contain gradient
    std::vector<Cell> cell_grads;
    size_t used_workspace_size = static_cast<uint8_t*>(workspace_ptr) -
                                 static_cast<uint8_t*>((void*)(workspace.raw_ptr));
    workspace_ptr =
            static_cast<uint8_t*>(workspace_ptr) +
            get_cells(
                    D, num_layers, input_size, hidden_size, bias, cell_grads, dw,
                    Workspace(
                            workspace.raw_ptr + used_workspace_size,
                            workspace.size - used_workspace_size));

    auto add_opr = handle->create_operator<ElemwiseForward>();
    add_opr->param().mode = Elemwise::Mode::ADD;
    auto copy_opr = handle->create_operator<TypeCvtForward>();

    // initialize dx to zero
    memset(dx.raw_ptr(), 0, dx.layout.span().dist_byte());

    // calculate grads
    for (int layer = num_layers - 1; layer >= 0; --layer) {
        for (int d = D - 1; d >= 0; --d) {
            Cell& cell = cells[layer * D + d];
            Cell& cell_grad = cell_grads[layer * D + d];
            size_t input_size = layer_inputs[layer].layout.shape[2];
            const TensorND& x_arr = layer_inputs[layer];
            const TensorND& y_arr = layer_outputs[layer];
            TensorLayout x_layout = {{batch_size, input_size}, dtype};

            // tmp tensors
            void* tmp_workspace_ptr = workspace_ptr;
            TensorND dwi_tmp{tmp_workspace_ptr, cell_grad.weight_ih.layout};
            tmp_workspace_ptr = static_cast<uint8_t*>(tmp_workspace_ptr) +
                                dwi_tmp.layout.span().dist_byte();
            TensorND dwh_tmp{tmp_workspace_ptr, cell_grad.weight_hh.layout};
            tmp_workspace_ptr = static_cast<uint8_t*>(tmp_workspace_ptr) +
                                dwh_tmp.layout.span().dist_byte();
            TensorND dbias_tmp{tmp_workspace_ptr, cell_grad.bias_ih.layout};
            tmp_workspace_ptr = static_cast<uint8_t*>(tmp_workspace_ptr) +
                                dbias_tmp.layout.span().dist_byte();
            size_t used_workspace_size =
                    static_cast<uint8_t*>(tmp_workspace_ptr) -
                    static_cast<uint8_t*>((void*)(workspace.raw_ptr));

            for (int i = 0; i < seq_len; ++i) {
                // reverse time step (not seq step). Here step means seq step
                size_t step = i;
                if (d == 0)
                    step = seq_len - i - 1;
                TensorND x{
                        static_cast<uint8_t*>(x_arr.raw_ptr()) +
                                step * x_layout.span().dist_byte(),
                        x_layout},
                        y{static_cast<uint8_t*>(y_arr.raw_ptr()) +
                                  (step * D + d) * cell_y_layout.span().dist_byte(),
                          cell_y_layout};
                const TensorNDArray& cell_states = cell_seq_states[layer * D + d][step];
                TensorNDArray& dstates_new = dstates_arr[layer * D + d];
                // dy should be d_output + d_hidden
                TensorND dy_t{
                        static_cast<uint8_t*>(layer_output_grad.raw_ptr()) +
                                (step * D + d) * cell_y_layout.span().dist_byte(),
                        cell_y_layout};
                add_opr->exec({dstates_new[0], dy_t}, dy_t);
                // dx for layer 0 has a different size
                TensorND dx_t;
                if (layer == 0)
                    dx_t = {static_cast<uint8_t*>(L0_direction_dx_arr[d].raw_ptr()) +
                                    step * x_layout.span().dist_byte(),
                            x_layout};
                else
                    dx_t = {static_cast<uint8_t*>(direction_dx_arr[d].raw_ptr()) +
                                    step * x_layout.span().dist_byte(),
                            x_layout};
                TensorNDArray douts = {dy_t};
                for (int s = 1; s < dstates_new.size(); ++s)
                    douts.push_back(dstates_new[s]);
                cell.backward(
                        handle, nonlineMode, x, cell_states, y, douts, dx_t,
                        dstates_new, dwi_tmp, dwh_tmp, dbias_tmp,
                        Workspace(
                                workspace.raw_ptr + used_workspace_size,
                                workspace.size - used_workspace_size));
                // add step gradient to overall gradient
                add_opr->exec({dwi_tmp, cell_grad.weight_ih}, cell_grad.weight_ih);
                add_opr->exec({dwh_tmp, cell_grad.weight_hh}, cell_grad.weight_hh);
                add_opr->exec({dbias_tmp, cell_grad.bias_ih}, cell_grad.bias_ih);
                add_opr->exec({dbias_tmp, cell_grad.bias_hh}, cell_grad.bias_hh);
            }
        }
        // add gradient of different directions to layer_output_grad.
        // Layer 0 to dx
        if (layer == 0) {
            for (int i = 0; i < D; ++i)
                add_opr->exec({L0_direction_dx_arr[i], dx}, dx);
        } else {
            if (D == 1)
                copy_opr->exec(direction_dx_arr[0], layer_output_grad);
            else {  // D == 2, arrange as [(d0, d1), (d0, d1), ...]
                for (size_t t = 0; t < seq_len; ++t) {
                    size_t offset = t * D * cell_y_layout.span().dist_byte();
                    for (size_t d = 0; d < D; ++d) {
                        TensorND src{
                                static_cast<uint8_t*>(direction_dx_arr[d].raw_ptr()) +
                                        offset,
                                cell_y_layout};
                        TensorND dst{
                                static_cast<uint8_t*>(layer_output_grad.raw_ptr()) +
                                        offset + d * cell_y_layout.span().dist_byte(),
                                cell_y_layout};
                        copy_opr->exec(src, dst);
                    }
                }
            }
        }
    }
}

}  // namespace rnn
}  // namespace naive
}  // namespace megdnn
