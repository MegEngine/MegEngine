#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace diag {

template <typename T>
void exec_internal_to_vector(
        T* src, T* dst, ptrdiff_t start, ptrdiff_t size, ptrdiff_t stride_sum,
        ptrdiff_t dst_stride, cudaStream_t stream);

template <typename T>
void exec_internal_to_matrix(
        T* src, T* dst, ptrdiff_t start, ptrdiff_t n, ptrdiff_t k,
        ptrdiff_t dst_stride0, ptrdiff_t dst_stride1, ptrdiff_t src_stride,
        cudaStream_t stream);

}  // namespace diag
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
