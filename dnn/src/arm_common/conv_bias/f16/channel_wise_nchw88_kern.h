#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {
namespace fp16 {
namespace channel_wise_nchw88 {

#define KERN(stride, i)                                                               \
    template <BiasMode bias_mode, typename Op>                                        \
    void do_conv_kern_##stride##_##i##x##i(                                           \
            const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst, \
            const size_t IH, const size_t IW, const size_t OH, const size_t OW,       \
            const size_t PH, const size_t PW);

KERN(stride1, 2)
KERN(stride1, 3)
KERN(stride1, 5)

KERN(stride2, 2)
KERN(stride2, 3)
KERN(stride2, 5)

#undef KERN

}  // namespace channel_wise_nchw88
}  // namespace fp16
}  // namespace arm_common
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
