#pragma once

#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

void sleep(cudaStream_t stream, uint64_t cycles);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
