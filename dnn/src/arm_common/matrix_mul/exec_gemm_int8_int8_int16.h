#pragma once
#include <cstddef>
#include <cstdint>

namespace megdnn {
namespace arm_common {

///! Row-major gemm
void exec_gemm_int8_int8_int16(
        const int8_t* A, const int8_t* B, int16_t* C, size_t M, size_t K, size_t N,
        size_t LDB, int8_t* w0, int8_t* w1);

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
