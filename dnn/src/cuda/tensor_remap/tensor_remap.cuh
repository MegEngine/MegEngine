#pragma once
#include "src/cuda/utils.cuh"

#include "megdnn/internal/defs.h"

namespace megdnn {
namespace cuda {
namespace tensor_remap {

template <typename ctype>
void forward(
        const ctype* src, const int* map, ctype* dst, uint32_t sdim, uint32_t ddim,
        const array_wrapper<int, MEGDNN_MAX_NDIM>& sstride,
        const array_wrapper<int, MEGDNN_MAX_NDIM>& dstride,
        const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& dshape, cudaStream_t stream);

template <typename ctype>
void backward(
        const ctype* diff, const int* map, ctype* grad, uint32_t sdim, uint32_t ddim,
        const array_wrapper<int, MEGDNN_MAX_NDIM>& sstride,
        const array_wrapper<int, MEGDNN_MAX_NDIM>& dstride,
        const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& sshape,
        const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& dshape, bool is_non_overlapping,
        cudaStream_t stream);

}  // namespace tensor_remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
