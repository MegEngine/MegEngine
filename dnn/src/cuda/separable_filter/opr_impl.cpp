#include "src/cuda/separable_filter/opr_impl.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cuda {

void SeparableFilterForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter_x, _megdnn_tensor_in filter_y,
        _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(
            src.layout, filter_x.layout, filter_y.layout, dst.layout, workspace.size);
    megdnn_assert(false, "SeparableFilter is not supported in CUDA");
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
