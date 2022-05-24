#pragma once

#include <cstddef>

namespace megdnn {
namespace fallback {

void gi_gemv_like_mk4(
        const float* __restrict A, const float* __restrict B, float* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
