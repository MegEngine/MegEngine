#include "megdnn/handle.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn.h"
#include "megdnn/oprs/nn_int.h"
#include "src/common/utils.h"

namespace megdnn {

void handle_bias_and_nonlinear(
        Handle* handle, param::ConvBias args, const TensorND* conv_dst_tensor,
        const TensorND* dst_tensor, const TensorND* bias_tensor);

}  // namespace megdnn

// vim: syntax=cpp.doxygen
