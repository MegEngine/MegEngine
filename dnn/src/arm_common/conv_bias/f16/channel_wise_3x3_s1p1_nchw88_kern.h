#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {
namespace fp16 {
namespace channel_wise_nchw88 {

template <BiasMode bias_mode, typename Op>
void do_conv_kern_3x3_stride1_padding1(
        const __fp16* src, __fp16* dst, const __fp16* filter, const __fp16* bias, int H,
        int W);

}  // namespace channel_wise_nchw88
}  // namespace fp16
}  // namespace arm_common
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
