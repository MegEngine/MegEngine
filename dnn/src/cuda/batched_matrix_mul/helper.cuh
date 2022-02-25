#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace batched_matrix_mul {

template <typename T>
void arange(T* Xs, T start, uint32_t step, uint32_t n, cudaStream_t stream);

}  // namespace batched_matrix_mul
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
