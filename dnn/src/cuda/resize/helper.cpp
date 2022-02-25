#include "helper.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cuda {
namespace resize {

InterpolationMode get_imode(param::Resize::InterpolationMode imode) {
    using IMode = param::Resize::InterpolationMode;
    switch (imode) {
        case IMode::NEAREST:
            return INTER_NEAREST;
        case IMode::LINEAR:
            return INTER_LINEAR;
        case IMode::AREA:
            return INTER_AREA;
        case IMode::CUBIC:
            return INTER_CUBIC;
        case IMode::LANCZOS4:
            return INTER_LANCZOS4;
        default:
            megdnn_throw("impossible");
    }
}

}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
