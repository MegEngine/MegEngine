#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {
namespace conv_bias {

template <size_t FH, size_t SH, BiasMode bias_mode, typename Op>
void conv_direct_fp16_nchw88(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        int IC, int IH, int IW, int OH, int OW);

}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn

#endif
