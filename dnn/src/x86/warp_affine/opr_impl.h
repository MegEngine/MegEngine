#pragma once
#include "src/naive/warp_affine/opr_impl.h"

namespace megdnn {
namespace x86 {

class WarpAffineImpl : public naive::WarpAffineImpl {
private:
    using naive::WarpAffineImpl::WarpAffineImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
