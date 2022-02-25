#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace linspace {

template <typename T>
void exec_internal(T* dst, double start, double step, size_t n, cudaStream_t stream);

}  // namespace linspace
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
