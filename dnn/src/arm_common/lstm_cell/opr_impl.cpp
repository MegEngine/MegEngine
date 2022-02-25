#include "src/arm_common/lstm_cell/opr_impl.h"
#include "src/common/lstm_cell.h"
#include "src/naive/handle.h"

#include "./cell_kernel.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_lstm_cell)

using namespace megdnn;
using namespace arm_common;

void LSTMCellImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_in cx, _megdnn_tensor_out h_new, _megdnn_tensor_out c_new,
        _megdnn_tensor_out gates, _megdnn_workspace workspace) {
    //! only float32 and {1, xx} shape bias will be optimized
    MIDOUT_BEGIN(megdnn_arm_common_lstm_cell, midout_iv(0)) {
        if (!LstmCellCompute::is_optimized(
                    input.layout, weight_ih.layout, bias_ih.layout, hx.layout,
                    weight_hh.layout, bias_hh.layout, cx.layout, h_new.layout,
                    c_new.layout, gates.layout)) {
            naive::LSTMCellImpl::exec(
                    input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new,
                    gates, workspace);
        } else {
            LstmCellCompute::run(
                    input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new,
                    gates, workspace, handle());
        }
    }
    MIDOUT_END();
}

size_t LSTMCellImpl::get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& cx, const TensorLayout& h_new, const TensorLayout& c_new,
        const TensorLayout& gates) {
    MIDOUT_BEGIN(megdnn_arm_common_lstm_cell, midout_iv(1)) {
        if (!LstmCellCompute::is_optimized(
                    input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new,
                    gates)) {
            return naive::LSTMCellImpl::get_workspace_in_bytes(
                    input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new,
                    gates);
        } else {
            return LstmCellCompute::get_workspace_bundle(
                           input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new,
                           c_new, gates)
                    .total_size_in_bytes();
        }
    }
    MIDOUT_END();
}

// vim: syntax=cpp.doxygen
