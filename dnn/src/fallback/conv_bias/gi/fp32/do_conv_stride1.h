#pragma once

#include <cstddef>

namespace megdnn {
namespace fallback {
namespace fp32 {
namespace conv_stride1 {

void do_conv_2x2_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_3x3_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_5x5_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
void do_conv_7x7_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC);
}  // namespace conv_stride1
}  // namespace fp32
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
