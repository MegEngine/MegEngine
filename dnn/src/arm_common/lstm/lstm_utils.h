#pragma once

#include "src/arm_common/lstm_cell/cell_kernel.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/lstm/opr_impl.h"

namespace megdnn {
namespace arm_common {

template <class CellOp, class States>
void cell_opr_compute(
        _megdnn_tensor_in step_input, _megdnn_tensor_in weight_ih,
        _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in bias_hh, const States& state_in, States& state_out,
        Workspace cell_workspace, Handle* handle);

struct LstmCellWeight {
    size_t m_weight_size = 0;

    TensorND m_weight_ih, m_weight_hh, m_bias_ih, m_bias_hh;
    // if no bias, will create dummy bias tensor from workspace
    LstmCellWeight(
            RefPtr weight_ptr, size_t hidden_size, size_t input_size, bool has_bias,
            DType dtype);
};

struct LstmStates {
    static size_t nr_states() { return 2; }
    size_t m_memory_size;
    TensorND m_h, m_c;
    LstmStates(
            const SmallVector<RefPtr> ptr, size_t hidden_size, size_t batch_size,
            DType dtype);
};

TensorNDArray split_tensor(
        _megdnn_tensor_in tensor, size_t nr_tensor, const TensorLayout& layout);

template <class CellWeight>
SmallVector<CellWeight> get_all_cells(
        size_t dir_size, size_t num_layers, size_t input_size, size_t hidden_size,
        bool bias, _megdnn_tensor_in flatten_weights) {
    SmallVector<CellWeight> cell_weights;
    cell_weights.reserve(dir_size * num_layers);
    auto weight_ptr = flatten_weights.get_ref_ptr();
    for (size_t layer = 0; layer < num_layers; ++layer) {
        for (size_t d = 0; d < dir_size; ++d) {
            size_t cell_input_size = layer == 0 ? input_size : dir_size * hidden_size;
            CellWeight cell_weight(
                    weight_ptr, hidden_size, cell_input_size, bias,
                    flatten_weights.layout.dtype);
            weight_ptr += cell_weight.m_weight_size;
            cell_weights.push_back(cell_weight);
        }
    }
    return cell_weights;
}

template <class States>
SmallVector<States> get_all_status(
        _megdnn_tensor_in hx, _megdnn_tensor_in cx, size_t hidden_size,
        size_t batch_size, size_t num_layers, size_t dir_size, DType dtype) {
    SmallVector<States> states;
    auto hx_ptr = hx.get_ref_ptr();
    auto cx_ptr = cx.get_ref_ptr();
    for (size_t layer = 0; layer < num_layers * dir_size; ++layer) {
        States state({hx_ptr, cx_ptr}, hidden_size, batch_size, dtype);
        hx_ptr += state.m_memory_size;
        cx_ptr += state.m_memory_size;
        states.push_back(state);
    }
    return states;
}

template <class Cell, typename CellOpr, class States>
void exec_kernel(
        SmallVector<Cell>& cells, const TensorNDArray& inputs,
        const SmallVector<States>& states_in, SmallVector<States>& states_out,
        TensorNDArray& outputs, size_t num_layers, size_t dir_size, Handle* handle,
        WorkspaceBundle workspace_bundle) {
    megdnn_assert(cells.size() == num_layers * dir_size);
    megdnn_assert(
            states_in.size() == states_out.size() &&
            states_in.size() == num_layers * dir_size);
    megdnn_assert(outputs.size() == inputs.size());
    //! two tmp state workspace
    megdnn_assert(workspace_bundle.nr_workspace() == 4 + States::nr_states());

    size_t seq_len = inputs.size();
    size_t batch_size = inputs[0].layout.shape[0];
    size_t input_size = inputs[0].layout.shape[1];
    size_t hidden_size = cells[0].m_weight_hh.layout.shape[1];

    TensorLayout batch_output_layout{
            {hidden_size}, outputs[0].layout.dtype};  // output hy
    TensorLayout cell_output_layout{
            {batch_size, hidden_size}, outputs[0].layout.dtype};  // output hy
    TensorLayout seq_output_layout{
            {batch_size, dir_size * hidden_size}, outputs[0].layout.dtype};
    TensorLayout cell_first_input_layout{
            {batch_size, input_size}, inputs[0].layout.dtype};  // input
    TensorLayout cell_input_layout{
            {batch_size, dir_size * hidden_size}, inputs[0].layout.dtype};
    TensorLayout tmp_output_layout{
            {seq_len, batch_size, dir_size * hidden_size}, outputs[0].layout.dtype};

    //! workspace get
    Workspace cell_workspace(
            static_cast<dt_byte*>(workspace_bundle.get(0)),
            workspace_bundle.get_size(0) + workspace_bundle.get_size(1));
    auto&& tmp_inputs_1 = split_tensor(
            TensorND{workspace_bundle.get(2), tmp_output_layout}, seq_len,
            cell_input_layout);
    auto&& tmp_outputs_1 = split_tensor(
            TensorND{workspace_bundle.get(2), tmp_output_layout}, seq_len,
            seq_output_layout);

    auto&& tmp_inputs_2 = split_tensor(
            TensorND{workspace_bundle.get(3), tmp_output_layout}, seq_len,
            cell_input_layout);
    auto&& tmp_outputs_2 = split_tensor(
            TensorND{workspace_bundle.get(3), tmp_output_layout}, seq_len,
            seq_output_layout);

    using IoPair = std::pair<TensorNDArray, TensorNDArray>;
    IoPair io_pair1 = {tmp_inputs_1, tmp_outputs_2};
    IoPair io_pair2 = {tmp_inputs_2, tmp_outputs_1};
    SmallVector<IoPair> io_pairs = {io_pair1, io_pair2};

    SmallVector<RefPtr> ptr;
    for (size_t index = 0; index < States::nr_states(); index++) {
        ptr.push_back(workspace_bundle.get(4 + index));
    }
    auto&& tmp_state = States(ptr, hidden_size, batch_size, outputs[0].layout.dtype);

    for (size_t layer = 0; layer < num_layers; layer++) {
        auto layer_inputs = io_pairs[layer % 2].first;
        auto layer_outputs = io_pairs[layer % 2].second;

        //! if last layer, direct write to output tensors
        if (num_layers - 1 == layer) {
            layer_outputs = outputs;
        }
        if (0 == layer) {
            layer_inputs = inputs;
        }
        for (size_t d = 0; d < dir_size; ++d) {
            size_t cell_idx = layer * dir_size + d;
            auto& cell = cells[cell_idx];
            auto& state_in_origin = states_in[cell_idx];
            auto& state_out_origin = states_out[cell_idx];

            auto state_in = state_in_origin;
            auto state_out = tmp_state;

            for (size_t i = 0; i < seq_len; ++i) {
                size_t step = d == 0 ? i : seq_len - 1 - i;
                auto& step_input = layer_inputs[step];
                auto& step_output = layer_outputs[step];

                if (i == seq_len - 1) {
                    state_out = state_out_origin;
                }
                //! task 1
                //! this CellOp will dispatch task inner, so here not dispatch task
                cell_opr_compute<CellOpr, LstmStates>(
                        step_input, cell.m_weight_ih, cell.m_weight_hh, cell.m_bias_ih,
                        cell.m_bias_hh, state_in, state_out, cell_workspace, handle);
                //! task 2
                //! copy output to continue space
                auto copy_to_output = [=]() {
                    //! if dir_size >1 and batch_size > 1, recorder to output
                    size_t stride = batch_output_layout.span().dist_byte();
                    if (dir_size > 1 && batch_size > 1) {
                        int8_t* source = static_cast<int8_t*>(state_out.m_h.raw_ptr());
                        int8_t* dst = static_cast<int8_t*>(step_output.raw_ptr()) +
                                      d * stride;
                        for (size_t b = 0; b < batch_size; b++) {
                            memcpy(dst, source, stride);
                            source += stride;
                            dst += dir_size * stride;
                        }
                    } else {
                        void* source = state_out.m_h.raw_ptr();
                        int8_t* dst = static_cast<int8_t*>(step_output.raw_ptr()) +
                                      d * stride;
                        memcpy(dst, source, state_out.m_h.layout.span().dist_byte());
                    }
                };
                MEGDNN_DISPATCH_CPU_KERN(
                        static_cast<naive::HandleImpl*>(handle), copy_to_output());

                //! state_in and state_out are read and write inplace
                if (0 == i) {
                    state_in = tmp_state;
                }
            }
        }
    }
}

template <typename CellOpr>
WorkspaceBundle get_workspace_bundle(
        const TensorLayout& input, const TensorLayout& output,
        const TensorLayout& flatten_weights, size_t hidden_size, size_t dir_size,
        size_t states_size) {
    size_t batch_size = input.shape[1];
    size_t input_size = input.shape[2];
    size_t gate_hidden_size = flatten_weights.shape[0];

    // cell workspace
    TensorLayout weight_ih{{gate_hidden_size, input_size}, flatten_weights.dtype};
    TensorLayout weight_hh{
            {gate_hidden_size, dir_size * hidden_size}, flatten_weights.dtype};
    TensorLayout bias{{1, gate_hidden_size}, flatten_weights.dtype};
    TensorLayout hx{{batch_size, dir_size * hidden_size}, input.dtype};

    auto cell_opr = inplace_cpu_handle()->create_operator<CellOpr>();

    TensorLayout h_new, c_new, gates;
    cell_opr->deduce_layout(
            input, weight_ih, bias, hx, weight_hh, bias, hx, h_new, c_new, gates);

    SmallVector<size_t> workspaces;
    //! the cell opr compute workspace
    size_t cell_opr_workspace = cell_opr->get_workspace_in_bytes(
            input, weight_ih, bias, hx, weight_hh, bias, hx, h_new, c_new, gates);
    workspaces.push_back(gates.span().dist_byte());
    workspaces.push_back(cell_opr_workspace);
    //! double tmp output memory
    size_t tmp_output_workspace = output.span().dist_byte();
    workspaces.push_back(tmp_output_workspace);
    workspaces.push_back(tmp_output_workspace);

    //! tmp states memory
    size_t tmp_state_workspace = hx.span().dist_byte();
    for (size_t i = 0; i < states_size; i++) {
        workspaces.push_back(tmp_state_workspace);
    }
    return {nullptr, workspaces};
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
