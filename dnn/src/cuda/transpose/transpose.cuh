#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace megdnn {
namespace cuda {

// (m, n) to (n, m)
template <typename T>
void transpose(
        const T* A, T* B, size_t m, size_t n, size_t LDA, size_t LDB,
        cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
