#pragma once

#include "megcore_cdefs.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace matrix_inverse {

void check_error(
        const int* src_info, uint32_t n, megcore::AsyncErrorInfo* dst_info,
        void* tracker, cudaStream_t stream);

}  // namespace matrix_inverse
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
