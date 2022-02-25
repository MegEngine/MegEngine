#pragma once
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>

namespace megdnn {
namespace cuda {
namespace param_pack {

template <typename T>
void concat_proxy(
        const T** srcs, T* dst, size_t srcs_size, size_t total_size,
        const int32_t* offsets, cudaStream_t stream);

}  // namespace param_pack
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
