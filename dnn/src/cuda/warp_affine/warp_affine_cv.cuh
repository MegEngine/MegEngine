#pragma once

#include <cstdio>
#include "src/common/cv/enums.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace warp_affine {

template <typename T, size_t CH>
void warp_affine_cv_proxy(
        const T* src, T* dst, const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols, const size_t src_step,
        const size_t dst_step, BorderMode bmode, InterpolationMode imode,
        const float* trans, T border_val, double* workspace, cudaStream_t stream);

}  // namespace warp_affine
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
