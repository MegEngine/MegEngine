#pragma once

#include <cstddef>
#include <cstdint>

namespace megdnn {
namespace arm_common {

bool is_gemv_like_preferred_int8(
        bool transposeA, bool transposeB, size_t M, size_t N, size_t K, size_t LDA,
        size_t LDB, size_t LDC);

void gemv_like(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);

void gemv_like_mk4(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);

#if MGB_ENABLE_DOT
void gemv_like_mk4_dot(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);
#endif

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
