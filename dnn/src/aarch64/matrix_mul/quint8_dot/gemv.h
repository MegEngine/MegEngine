#pragma once

#include "src/common/utils.h"

#if MGB_ENABLE_DOT
namespace megdnn {
namespace aarch64 {
namespace matmul {

bool is_gemv_like_preferred_quint8(
        bool transposeA, bool transposeB, size_t M, size_t N, size_t K, size_t LDA,
        size_t LDB, size_t LDC);

void gemv_like_quint8(
        const uint8_t* __restrict A, const uint8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride,
        uint8_t zero_point_A, uint8_t zero_point_B);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
