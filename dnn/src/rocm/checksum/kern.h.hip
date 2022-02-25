#pragma once

#include "hip_header.h"

namespace megdnn {
namespace rocm {
namespace checksum {

void calc(uint32_t* dest, const uint32_t* buf, uint32_t* workspace,
          size_t nr_elem, hipStream_t stream);

size_t get_workspace_in_bytes(size_t nr_elem);

}  // namespace checksum
}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

