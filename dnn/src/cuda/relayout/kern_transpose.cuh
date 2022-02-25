#pragma once

#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
template <typename T>
void copy_by_transpose(
        const T* A, T* B, size_t batch, size_t m, size_t n, size_t lda, size_t ldb,
        size_t stride_a, size_t stride_b, cudaStream_t stream);
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
