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

#include "midout.h"
MIDOUT_DECL(megdnn_naive_rnn_fwd)

namespace megdnn {
namespace naive {

using rnn::RNNCellWeightWrapper;

void RNNImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in hx,
        _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
        _megdnn_tensor_out hy, _megdnn_tensor_out reserve_space,
        _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_rnn_fwd) {
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
                _param.nonlineMode, this->handle(), new_workspace);
    }
    MIDOUT_END();
}

size_t RNNImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& hx,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout& /*hy*/, const TensorLayout& /*reserve_space*/) {
    auto _param = param();
    size_t D = _param.bidirectional ? 2 : 1;
    size_t last_dim = std::max(input.shape[2], D * hx.shape[1]);
    TensorLayout last_input = {{input.shape[0], input.shape[1], last_dim}, input.dtype};
    size_t workspace_size = rnn::get_workspace_in_bytes<RNNCellForward>(
            last_input, flatten_weights, param().hidden_size,
            param().bidirectional ? 2 : 1, this->handle());
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
    TensorNDArray layer_outputs;
    std::vector<std::vector<TensorNDArray>> cell_seq_states;
    size_t num_layers = param().num_layers;
    size_t D = param().bidirectional ? 2 : 1;
    size_t input_size = x.layout.shape[2];
    size_t batch_size = x.layout.shape[1];
    size_t hidden_size = param().hidden_size;
    size_t used_workspace_size = 0;

    // get cells
    std::vector<RNNCellWeightWrapper> cells;
    used_workspace_size += rnn::get_cells(
            D, num_layers, input_size, hidden_size, param().bias, cells,
            flatten_weights, workspace);

    // nonlinear mode
    param::RNNCell::NonlineMode nonlineMode = param::RNNCell::NonlineMode::TANH;
    using ModeRNN = param::RNN::NonlineMode;
    using ModeRNNCell = param::RNNCell::NonlineMode;
    switch (param().nonlineMode) {
        case ModeRNN::RELU:
            nonlineMode = ModeRNNCell::RELU;
            break;
        case ModeRNN::TANH:
            nonlineMode = ModeRNNCell::TANH;
            break;
        case ModeRNN::IDENTITY:
            break;
    }

    // get formatted inputs
    Workspace new_workspace = Workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    TensorLayout unfold_hx_layout{
            TensorShape{batch_size, hidden_size}, hx.layout.dtype};
    std::vector<TensorNDArray> hx_param;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        for (size_t d = 0; d < D; ++d) {
            TensorNDArray unfold_hx;
            size_t idx = layer * D + d;
            size_t states_offset = idx * unfold_hx_layout.span().dist_byte();
            unfold_hx.push_back(TensorND{
                    static_cast<uint8_t*>(hx.raw_ptr()) + states_offset,
                    unfold_hx_layout});
            hx_param.push_back(unfold_hx);
        }
    }
    used_workspace_size += rnn::get_inputs_for_exec<RNNCellWeightWrapper>(
            x, y, hx_param, reserve_space, num_layers, D, hidden_size, cells,
            layer_inputs, layer_outputs, cell_seq_states, nonlineMode, new_workspace);

    TensorNDArray dhy_arr = {dhy}, dhx_arr = {dhx};

    new_workspace = Workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    rnn::backward_exec_internal<RNNCellWeightWrapper>(
            cells, D, num_layers, input_size, param().bias, nonlineMode, layer_inputs,
            layer_outputs, cell_seq_states, dy, dhy_arr, dx, dhx_arr, dw,
            this->handle(), new_workspace);
}

size_t RNNBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout& y, const TensorLayout& /*hx*/,
        const TensorLayout& /*dy*/, const TensorLayout& /*dhy*/,
        const TensorLayout& flatten_weights, const TensorLayout& /*reserve_space*/,
        const TensorLayout& /*dx*/, const TensorLayout& /*dhx*/,
        const TensorLayout& /*dw*/) {
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
