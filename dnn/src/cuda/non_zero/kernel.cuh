#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace non_zero {
void expansion_index(
        dt_int32* dst_pt, size_t index_size, const size_t* src_shape,
        size_t* src_shape_workspace_pt, size_t src_ndim, dt_int32* div_workspace_pt,
        cudaStream_t stream);

void copy_idx(
        dt_int32* dest_idx, dt_int32* src_idx, uint32_t size, cudaStream_t stream);
}  // namespace non_zero
}  // namespace cuda
}  // namespace megdnn