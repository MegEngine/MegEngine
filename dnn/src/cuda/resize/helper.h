#pragma once
#include "src/cuda/resize/common.h"

#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {
namespace resize {

InterpolationMode get_imode(param::Resize::InterpolationMode imode);

}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
