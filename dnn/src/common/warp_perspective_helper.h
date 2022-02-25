#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace warp_perspective {
using Param = param::WarpPerspective;
bool is_cv_available(
        const TensorLayout& src, const TensorLayout& mat, const TensorLayout& mat_idx,
        const TensorLayout& dst, Param param);
bool is_dnn_available(
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, Param param);
}  // namespace warp_perspective
}  // namespace megdnn

// vim: syntax=cpp.doxygen
