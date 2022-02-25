#pragma once

#include "src/arm_common/convolution/opr_impl.h"
#if MGB_ENABLE_DOT

namespace megdnn {
namespace arm_common {
namespace deconv {

using NCBKernSizeParam = ConvolutionBackwardDataImpl::NCBKernSizeParam;
using NCBKernParam = ConvolutionBackwardDataImpl::NCBKernParam;

bool can_stride1_quint8_dot(const NCBKernSizeParam& param);

void stride1_quint8_dot(const NCBKernParam& param);

size_t get_workspace_in_bytes_stride1_quint8_dot(const NCBKernSizeParam& param);

}  // namespace deconv
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
