#include "src/naive/lstm_cell/opr_impl.h"
#include "src/common/lstm_cell.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_lstmcell_fwd)

namespace megdnn {
namespace naive {
size_t LSTMCellImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& cx, const TensorLayout& h_new, const TensorLayout& c_new,
        const TensorLayout& gates) {
    return megdnn::lstm_cell::get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new, gates,
            handle());
}

void LSTMCellImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_in cx, _megdnn_tensor_out h_new, _megdnn_tensor_out c_new,
        _megdnn_tensor_out gates, _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_lstmcell_fwd) {
        megdnn::lstm_cell::exec(
                input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new,
                gates, workspace, handle());
    }
    MIDOUT_END();
}
}  // namespace naive

}  // namespace megdnn