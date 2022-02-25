#pragma once

#include "./opr_impl.h"

namespace megdnn {
namespace fallback {
namespace convolution {

void run_conv(
        const float* src, const float* filter, float* dst, void* workspace, size_t IH,
        size_t IW, size_t IC, size_t FH, size_t FW, size_t OH, size_t OW, size_t OC,
        size_t PH, size_t PW, size_t SH, size_t SW, bool xcorr);

void run_conv_backward_data(
        const float* diff, const float* filter, float* grad, void* workspace, size_t IH,
        size_t IW, size_t IC, size_t FH, size_t FW, size_t OH, size_t OW, size_t OC,
        size_t PH, size_t PW, size_t SH, size_t SW, bool xcorr);

}  // namespace convolution
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
