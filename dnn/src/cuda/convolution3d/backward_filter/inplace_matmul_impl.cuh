#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace convolution3d {

void exec_inplace_matmul_bwd_filter(
        const float* diff, const float* src, float* grad, size_t N, size_t INP_BS,
        size_t OUT_BS, size_t IC, size_t ID, size_t IH, size_t IW, size_t OC, size_t OD,
        size_t OH, size_t OW, size_t FD, size_t FH, size_t FW, size_t PD, size_t PH,
        size_t PW, size_t SD, size_t SH, size_t SW, size_t DD, size_t DH, size_t DW,
        bool is_xcorr, cudaStream_t stream);

}  // namespace convolution3d
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
