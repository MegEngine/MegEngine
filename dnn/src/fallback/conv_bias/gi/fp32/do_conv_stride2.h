#pragma once

#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace fallback {
namespace fp32 {
namespace conv_stride2 {
void do_conv_2x2_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_3x3_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_5x5_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_7x7_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
}  // namespace conv_stride2
}  // namespace fp32
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
