#include <megdnn/oprs.h>

#include "src/common/cv/helper.h"
#include "src/fallback/resize/gi/helper.h"

namespace megdnn {
namespace fallback {

/**
 * \fn resize_cv_exec
 * \brief Used if the format is NHWC, transfer from megcv
 */
void resize_cv_gi_exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst,
        param::Resize::InterpolationMode imode);

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
