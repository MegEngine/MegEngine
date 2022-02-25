#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace fill {

template <typename T>
void exec_internal(T* dst, T value, size_t size, cudaStream_t stream);

}  // namespace fill
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
