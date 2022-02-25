#pragma once
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace split {

template <typename T>
void forward_proxy(
        const T* src, T** dsts, size_t nr_dsts, size_t A, size_t B, size_t C,
        const size_t* Bv, const size_t* table_outer, const size_t* table_inner,
        cudaStream_t stream);

}  // namespace split
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
