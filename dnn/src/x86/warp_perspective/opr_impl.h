#pragma once
#include "megdnn/oprs.h"
#include "src/fallback/warp_perspective/opr_impl.h"

namespace megdnn {
namespace x86 {

class WarpPerspectiveImpl : public fallback::WarpPerspectiveImpl {
public:
    using fallback::WarpPerspectiveImpl::WarpPerspectiveImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
