#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {

void im2col_nhwc_int8(
        const int8_t* src, int8_t* unrolled, uint32_t N, uint32_t IH, uint32_t IW,
        uint32_t IC, uint32_t IWS, uint32_t OH, uint32_t OW, uint32_t OC, uint32_t OWS,
        uint32_t FH, uint32_t FW, uint32_t PH, uint32_t PW, uint32_t SH, uint32_t SW,
        uint32_t DH, uint32_t DW, uint32_t LD, bool flip, cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
