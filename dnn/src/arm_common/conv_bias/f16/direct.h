#pragma once

#include <cstddef>
#include "megdnn/dtype.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {
namespace fp16 {
namespace conv_bias {

void kern_direct_f16(
        const __fp16* src, const __fp16* filter, __fp16* dst, const int IH,
        const int IW, const int OH, const int OW, const int FH, const int FW);

}  // namespace conv_bias
}  // namespace fp16
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
