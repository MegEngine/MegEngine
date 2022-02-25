#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace repeat {

template <typename T>
void forward_proxy(
        const T* src, T* dst, size_t ndim, const size_t* sshape, const size_t* dshape,
        const size_t* tshape, cudaStream_t stream);

}  // namespace repeat
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
