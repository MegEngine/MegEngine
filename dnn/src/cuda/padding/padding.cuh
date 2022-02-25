#pragma once
#include "cuda_runtime.h"
#include "megdnn/basic_types.h"
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace padding {

template <typename T>
void padding_forward_proxy(
        const TensorND& src, const TensorND& dst, size_t offsets[MEGDNN_MAX_NDIM * 2],
        uint32_t mode, const float_t padding_val, cudaStream_t stream);

template <typename T>
void padding_backward_proxy(
        const TensorND& src, const TensorND& dst, size_t offsets[MEGDNN_MAX_NDIM * 2],
        uint32_t mode, cudaStream_t stream);

}  // namespace padding
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen