#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace cumprod {

void get_BX_BY(uint32_t A, uint32_t B, uint32_t C, uint32_t& BX, uint32_t& BY);

uint32_t get_workspace_bytes_for_cub_1d(uint32_t nr_item, uint32_t item_size);

}  // namespace cumprod
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
