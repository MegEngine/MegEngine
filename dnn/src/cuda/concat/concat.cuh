#pragma once
#include <stdint.h>
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace concat {

template <typename T>
void forward_proxy(
        const T** srcs, T* dst, size_t nr_srcs, size_t A, size_t B, size_t C,
        const size_t* Bv, const size_t* table_outer, const size_t* table_inner,
        cudaStream_t stream);

}  // namespace concat
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
