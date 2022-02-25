#pragma once

#include "hip_header.h"
#include "megdnn/basic_types.h"

namespace megdnn {
namespace rocm {

void copy_noncontig_general(const TensorND& dst, const TensorND& src,
                            hipStream_t stream);
}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen


