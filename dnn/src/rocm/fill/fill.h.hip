#pragma once
#include <stdint.h>
#include "hip_header.h"

namespace megdnn {
namespace rocm {
namespace fill {

template <typename T>
void exec_internal(T *dst, T value, size_t size, hipStream_t stream);

} // namespace fill
} // namespace rocm
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
