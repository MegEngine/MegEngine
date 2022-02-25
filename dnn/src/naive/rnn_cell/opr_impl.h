#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class RNNCellImpl : public RNNCell {
public:
    using RNNCell::RNNCell;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
            _megdnn_tensor_in bias_ih, _megdnn_tensor_in hx,
            _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& dst) override;
    bool is_thread_safe() const override { return true; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
