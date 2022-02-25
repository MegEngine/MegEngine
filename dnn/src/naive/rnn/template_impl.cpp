#include "src/naive/rnn/funcs.h"

namespace megdnn {
namespace naive {
namespace rnn {

template <>
void cell_opr_exec<RNNCellForward>(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
        _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in bias_hh, const TensorNDArray& states,
        TensorNDArray& states_new, _megdnn_workspace workspace,
        param::RNNCell::NonlineMode nonline_mode, Handle* handle) {
    auto opr = handle->create_operator<RNNCellForward>();
    opr->param().nonlineMode = nonline_mode;
    opr->exec(
            input, weight_ih, bias_ih, states[0], weight_hh, bias_hh, states_new[0],
            workspace);
}

template <>
size_t cell_opr_get_workspace_in_bytes<RNNCellForward>(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& weight_hh, const TensorLayout& bias_ih,
        const TensorLayout& bias_hh, const TensorLayout& hx, Handle* handle) {
    auto cell_opr = handle->create_operator<RNNCellForward>();
    return cell_opr->get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, hx);
}

}  // namespace rnn
}  // namespace naive
}  // namespace megdnn