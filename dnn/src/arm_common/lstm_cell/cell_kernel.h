#pragma once

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/lstm_cell/opr_impl.h"

namespace megdnn {
namespace arm_common {

struct LstmCellCompute {
    static WorkspaceBundle get_workspace_bundle(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& cx, const TensorLayout& h_new,
            const TensorLayout& c_new, const TensorLayout& gates);

    static bool is_optimized(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& cx, const TensorLayout& h_new,
            const TensorLayout& c_new, const TensorLayout& gates);

    static void run(
            _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
            _megdnn_tensor_in bias_ih, _megdnn_tensor_in hx,
            _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
            _megdnn_tensor_in cx, _megdnn_tensor_out h_new, _megdnn_tensor_out c_new,
            _megdnn_tensor_out gates, _megdnn_workspace workspace, Handle* handle);
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
