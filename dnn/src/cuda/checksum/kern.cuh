#pragma once

#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace checksum {

void calc(
        uint32_t* dest, const uint32_t* buf, uint32_t* workspace, size_t nr_elem,
        cudaStream_t stream);

size_t get_workspace_in_bytes(size_t nr_elem);

}  // namespace checksum
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
