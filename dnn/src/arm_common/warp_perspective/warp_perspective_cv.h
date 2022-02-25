#include <megdnn/oprs.h>

#include "src/common/cv/helper.h"

namespace megdnn {
namespace arm_common {

/**
 * \fn warp_perspective_cv
 * \brief Used if the format is NHWC, transfer from megcv
 */
void warp_perspective_cv_exec(
        _megdnn_tensor_in src, _megdnn_tensor_in trans, _megdnn_tensor_in mat_idx,
        _megdnn_tensor_in dst, float border_value,
        param::WarpPerspective::BorderMode border_mode,
        param::WarpPerspective::InterpolationMode imode, Handle* handle);

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
