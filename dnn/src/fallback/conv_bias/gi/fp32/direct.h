#pragma once

#include <cstddef>

namespace megdnn {
namespace fallback {
namespace fp32 {
namespace conv_bias {

void kern_direct(
        const float* src, const float* filter, float* dst, const int IH, const int IW,
        const int OH, const int OW, const int FH, const int FW);

}  // namespace conv_bias
}  // namespace fp32
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
