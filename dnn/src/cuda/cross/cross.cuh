#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace cross {

template <typename T>
void exec_internal(
        T* A, size_t stride_a0, size_t stride_a1, T* B, size_t stride_b0,
        size_t stride_b1, T* C, size_t stride_c0, size_t stride_c1, size_t N,
        cudaStream_t stream);

}  // namespace cross
}  // namespace cuda
}  // namespace megdnn
   // vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}