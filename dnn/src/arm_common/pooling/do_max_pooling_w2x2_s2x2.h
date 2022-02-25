#pragma once
#include "src/fallback/pooling/opr_impl.h"

namespace megdnn {
namespace arm_common {
void pooling_max_w2x2_s2x2(
        const int8_t* src, int8_t* dst, size_t N, size_t C, size_t IH, size_t IW,
        size_t OH, size_t OW);
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
