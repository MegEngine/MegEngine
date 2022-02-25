#pragma once
#include "megdnn/oprs/base.h"
#include "megdnn/oprs/general.h"

namespace megdnn {
namespace rnn_cell {

size_t get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& dst, Handle* handle);

void exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_out dst, _megdnn_workspace workspace,
        param::RNNCell::NonlineMode nonline_mode, Handle* handle);

}  // namespace rnn_cell
}  // namespace megdnn