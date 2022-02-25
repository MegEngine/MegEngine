#include <megdnn/oprs.h>

#include "src/common/cv/helper.h"

namespace megdnn {
namespace x86 {

/**
 * \fn warp_affine_cv
 * \brief Used if the format is NHWC, transfer from megcv
 */
void warp_affine_cv_exec(
        _megdnn_tensor_in src, _megdnn_tensor_in trans, _megdnn_tensor_in dst,
        float border_value, param::WarpAffine::BorderMode border_mode,
        param::WarpAffine::InterpolationMode imode, Handle* handle);

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
