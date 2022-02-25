#pragma once
#include "src/common/opr_delegate.h"
#include "src/naive/rnn_cell/opr_impl.h"

namespace megdnn {
namespace arm_common {

class RNNCellImpl : public naive::RNNCellImpl {
public:
    using naive::RNNCellImpl::RNNCellImpl;
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

private:
    WorkspaceBundle get_workspace_bundle(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& dst);
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
