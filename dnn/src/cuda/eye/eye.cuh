#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace eye {

template <typename T>
void exec_internal(T* dst, size_t m, size_t n, int k, cudaStream_t stream);

}  // namespace eye
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
