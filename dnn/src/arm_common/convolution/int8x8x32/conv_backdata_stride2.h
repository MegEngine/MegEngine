#pragma once

#include "src/arm_common/convolution/opr_impl.h"
#if MGB_ENABLE_DOT

#include <cstddef>
#include <cstdint>

namespace megdnn {
namespace arm_common {
namespace deconv {

using NCBKernSizeParam = ConvolutionBackwardDataImpl::NCBKernSizeParam;
using NCBKernParam = ConvolutionBackwardDataImpl::NCBKernParam;

bool can_stride2_int8x8x32_dot(const NCBKernSizeParam& param);

void stride2_int8x8x32_dot(const NCBKernParam& param);

size_t get_workspace_in_bytes_stride2_int8x8x32_dot(const NCBKernSizeParam& param);

}  // namespace deconv
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
