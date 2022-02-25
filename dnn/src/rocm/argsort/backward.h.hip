#pragma once

#include "hip_header.h"
#include <stdint.h>

namespace megdnn {
namespace rocm {
namespace argsort {

template <typename T>
void backward_proxy(uint32_t dst_h, uint32_t dst_w, uint32_t src_w, T* dst,
                    const T* src_data, const int* src_idx, hipStream_t stream);

}  // namespace argsort
}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen

