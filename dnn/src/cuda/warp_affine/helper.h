#pragma once

#include "megdnn/oprs.h"
#include "src/common/cv/enums.h"

namespace megdnn {
namespace cuda {
namespace warp_affine {

BorderMode get_bmode(param::WarpAffine::BorderMode bmode);
InterpolationMode get_imode(param::WarpAffine::InterpolationMode imode);

}  // namespace warp_affine
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
