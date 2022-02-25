#pragma once
#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace dot {

template <typename T>
void run(
        const T* a, const T* b, T* c, float* workspace, uint32_t n, int32_t strideA,
        int32_t strideB, cudaStream_t stream);

}  // namespace dot
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
