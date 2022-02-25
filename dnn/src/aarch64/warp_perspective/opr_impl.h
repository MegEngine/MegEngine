#pragma once
#include "megdnn/oprs.h"
#include "src/arm_common/warp_perspective/opr_impl.h"

namespace megdnn {
namespace aarch64 {

class WarpPerspectiveImpl : public arm_common::WarpPerspectiveImpl {
public:
    using arm_common::WarpPerspectiveImpl::WarpPerspectiveImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
