#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace where {

template <typename T>
void forward_proxy(
        const bool* __restrict mask, const T* __restrict data1,
        const T* __restrict data2, T* __restrict dst, size_t n, cudaStream_t stream);

}  // namespace where

namespace where_backward {

template <typename T>
void backward_proxy(
        const T* __restrict diff, const bool* mask, T* __restrict grad_data1,
        T* __restrict grad_data2, size_t n, cudaStream_t stream);

}  // namespace where_backward

}  // namespace cuda
}  // namespace megdnn
