#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace deconv {

void reorder_filter_nc4hw4_to_n4hwc4(
        int8_t* dst, const int8_t* src, uint32_t OC, uint32_t IC, uint32_t FH,
        uint32_t FW, cudaStream_t stream);

void reorder_filter_nhwc_to_cnxhwx(
        int8_t* dst, const int8_t* src, uint32_t OC, uint32_t IC, uint32_t FH,
        uint32_t FW, uint32_t interleaved, cudaStream_t stream);

}  // namespace deconv
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
