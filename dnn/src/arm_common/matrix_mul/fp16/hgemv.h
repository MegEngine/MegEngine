#pragma once

#include <stddef.h>
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {

bool is_hgemv_preferred(
        bool transposeA, bool transposeB, size_t M, size_t N, size_t K, size_t /*LDA*/,
        size_t LDB, size_t /*LDC*/);

void gemv_like(
        const __fp16* __restrict A, const __fp16* __restrict B, __fp16* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);

}  // namespace arm_common
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
