#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace cutlass_wrapper {

template <uint32_t size_bits, uint32_t interleaved>
void reorder_ncxhwx_imma_filter(
        int8_t* dst_filter, const int8_t* src_filter, uint32_t OC, uint32_t IC,
        uint32_t FH, uint32_t FW, bool trans_oc, cudaStream_t stream);

template <uint32_t size_bits>
void reorder_nhwc_imma_filter(
        int8_t* dst_filter, const int8_t* src_filter, uint32_t OC, uint32_t IC,
        uint32_t FH, uint32_t FW, bool trans_oc, uint32_t alignbits,
        uint32_t interleaved, cudaStream_t stream);
}  // namespace cutlass_wrapper
}  // namespace cuda
}  // namespace megdnn
