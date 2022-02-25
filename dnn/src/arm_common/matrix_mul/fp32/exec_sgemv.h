#pragma once

#include <cstddef>

namespace megdnn {
namespace arm_common {

bool is_sgemv_like_preferred(
        bool row_major, bool transposeA, bool transposeB, size_t M, size_t N, size_t K,
        float alpha, size_t /* LDA */, size_t LDB, float beta, size_t /* LDC */);

void gemv_like(
        const float* __restrict A, const float* __restrict B, float* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);

void gemv_like_mk4(
        const float* __restrict A, const float* __restrict B, float* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
