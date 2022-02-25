#include "src/cuda/max_tensor_diff/opr_impl.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace cuda;

float MaxTensorDiffImpl::exec(_megdnn_tensor_in, _megdnn_tensor_in, _megdnn_workspace) {
    megdnn_throw("MaxTensorDiff not support in cuda");
}

// vim: syntax=cpp.doxygen
