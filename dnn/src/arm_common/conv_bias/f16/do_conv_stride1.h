#pragma once

#include "src/fallback/conv_bias/opr_impl.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace megdnn {
namespace arm_common {
namespace fp16 {
namespace conv_stride1 {
void do_conv_2x2_stride1(
        const __fp16* src, const __fp16* filter, __fp16* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_3x3_stride1(
        const __fp16* src, const __fp16* filter, __fp16* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_5x5_stride1(
        const __fp16* src, const __fp16* filter, __fp16* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
}  // namespace conv_stride1
}  // namespace fp16
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
