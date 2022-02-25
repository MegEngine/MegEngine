#include <megdnn/oprs.h>

#include "src/common/cv/helper.h"

namespace megdnn {
namespace x86 {

/**
 * \fn resize_cv_exec
 * \brief Used if the format is NHWC, transfer from megcv
 */
void resize_cv_exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst,
        param::Resize::InterpolationMode imode);

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
