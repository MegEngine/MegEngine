#pragma once

#include "hip_header.h"
#include "src/rocm/utils.h.hip"

namespace megdnn {
namespace rocm {

    void sleep(hipStream_t stream, uint64_t cycles);

} // namespace rocm
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
