#include "src/aarch64/warp_perspective/opr_impl.h"

#include "src/aarch64/warp_perspective/warp_perspective_cv.h"

#include "src/common/utils.h"
#include "src/common/warp_common.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace aarch64 {

void WarpPerspectiveImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
        _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, mat.layout, mat_idx.layout, dst.layout, workspace.size);
    if (warp::is_cv_available(
                src.layout, mat.layout, dst.layout, param().imode, param().format)) {
        warp_perspective_cv_exec(
                src, mat, mat_idx, dst, param().border_val, param().bmode,
                param().imode, handle());
    } else {
        //! Use arm_common implementation
        arm_common::WarpPerspectiveImpl::exec(src, mat, mat_idx, dst, workspace);
    }
}

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
