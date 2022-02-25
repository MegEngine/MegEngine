#pragma once

#include "src/common/utils.h"
#include "src/naive/lstm/opr_impl.h"

namespace megdnn {
namespace arm_common {

class LSTMImpl : public naive::LSTMImpl {
public:
    using naive::LSTMImpl::LSTMImpl;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in hx, _megdnn_tensor_in cx,
            _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
            _megdnn_tensor_out hy, _megdnn_tensor_out cy,
            _megdnn_tensor_out reserve_space, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
            const TensorLayout& flatten_weights, const TensorLayout& output,
            const TensorLayout& hy, const TensorLayout& cy,
            const TensorLayout& reserve_space) override;

    //! in arm_common only store the output tensor, other tensor is only
    //! used in computing grad, so arm ignore them
    size_t get_reserve_size_in_bytes(const TensorLayout&) override { return 1; }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
