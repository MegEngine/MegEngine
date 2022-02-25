#pragma once

#include "hip_header.h"
#include <stdint.h>
#include <stdio.h>

namespace megdnn {
namespace rocm {
namespace param_pack {

template <typename T>
void concat_proxy(const T** srcs, T* dst, size_t srcs_size, size_t total_size,
                  const int32_t* offsets, hipStream_t stream);

}  // namespace param_pack
}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
