#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace megdnn {
namespace cuda {
namespace flip {

template <typename T, bool vertical, bool horizontal>
void flip(
        const T* src, T* dst, size_t N, size_t H, size_t W, size_t IC, size_t stride1,
        size_t stride2, size_t stride3, cudaStream_t stream);

}  // namespace flip
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
