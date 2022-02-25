#pragma once
#include <cstddef>
#include <cstdint>
#include "src/arm_common/simd_macro/marm_neon.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "src/common/utils.h"

namespace megdnn {
namespace arm_common {

void do_max_pooling_3x3_s2x2_float16_NEON(
        const __fp16* src, __fp16* dst, size_t IH, size_t IW, size_t OH, size_t OW,
        size_t PH, size_t PW, const WorkspaceBundle& ws);

}  // namespace arm_common
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
