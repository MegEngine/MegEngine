#include "src/cuda/separable_conv/opr_impl.h"
#include <cstring>
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace cuda {
// using namespace sep_conv;

void SeparableConvForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter_x, _megdnn_tensor_in filter_y,
        _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(
            src.layout, filter_x.layout, filter_y.layout, dst.layout, workspace.size);
    megdnn_assert(false, "SeparableConv is not supported in CUDA");
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
