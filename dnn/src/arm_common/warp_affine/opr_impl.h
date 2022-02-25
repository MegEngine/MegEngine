#pragma once
#include "src/naive/warp_affine/opr_impl.h"

namespace megdnn {
namespace arm_common {

class WarpAffineImpl : public naive::WarpAffineImpl {
public:
    using naive::WarpAffineImpl::WarpAffineImpl;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
