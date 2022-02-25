#pragma once

#include <cstddef>
#include "megdnn/arch.h"
#include "src/common/utils.h"

namespace megdnn {
namespace aarch64 {

MEGDNN_NOINLINE void sgemm_packA_n(
        const float* A, float* Apacked, size_t M, size_t K, size_t LDA,
        const float* alpha);

MEGDNN_NOINLINE void sgemm_packA_t(
        const float* A, float* Apacked, size_t M, size_t K, size_t LDA,
        const float* alpha);

MEGDNN_NOINLINE void sgemm_packB_n(
        const float* B, float* Bpacked, size_t K, size_t N, size_t LDB);

MEGDNN_NOINLINE void sgemm_packB_t(
        const float* B, float* Bpacked, size_t K, size_t N, size_t LDB);

MEGDNN_NOINLINE void sgemm_kernel12x8(
        const float* A, const float* B, float* C, size_t LDC, size_t M, size_t N,
        size_t K, int type, const float* beta);

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
