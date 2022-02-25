#pragma once
#include "megdnn/oprs.h"
#include "src/fallback/resize/opr_impl.h"

namespace megdnn {
namespace arm_common {
class ResizeImpl : public fallback::ResizeImpl {
public:
    using fallback::ResizeImpl::ResizeImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
