#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace argsort {

template <typename T>
void backward_proxy(
        uint32_t dst_h, uint32_t dst_w, uint32_t src_w, T* dst, const T* src_data,
        const int* src_idx, cudaStream_t stream);

}  // namespace argsort
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
