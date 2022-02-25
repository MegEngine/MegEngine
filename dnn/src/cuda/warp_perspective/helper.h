#pragma once
#include "src/cuda/warp_perspective/common.h"

#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {
namespace warp_perspective {

BorderMode get_bmode(param::WarpPerspective::BorderMode bmode);
InterpolationMode get_imode(param::WarpPerspective::InterpolationMode imode);

}  // namespace warp_perspective
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
