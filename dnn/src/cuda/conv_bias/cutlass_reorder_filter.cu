/**
 * \file dnn/src/cuda/conv_bias/cutlass_reorder_filter.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/conv_bias/cutlass_reorder_filter.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/integer_subbyte_utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace cutlass_wrapper;

namespace {
template <uint32_t size_bits, uint32_t interleaved>
__device__ __forceinline__ void reorder_ncxhwx_imma_filter_func(
        int8_t* dst, const int8_t* src, uint32_t OC, uint32_t IC, uint32_t FH,
        uint32_t FW, uint32_t lane, bool trans_oc) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    static constexpr uint32_t threads_per_interleaved =
            interleaved / elements_per_lane;
    static constexpr uint32_t instruction_shape_col = 8;
    // 4 threads per Quad
    static constexpr uint32_t elements_per_thread = instruction_shape_col / 4;
    // 4 threads per Quad
    static constexpr uint32_t reordered_elements_per_thread = interleaved / 4;

    uint32_t id = lane / threads_per_interleaved;
    uint32_t residue = lane % threads_per_interleaved;
    uint32_t ICx = IC / interleaved;
    uint32_t row = id / (ICx * FH * FW);
    uint32_t col = id - row * ICx * FH * FW;
    // transpose ncxhwx to cxhwnx
    uint32_t src_offset = id * interleaved + residue * elements_per_lane;

    row = (trans_oc) ? (row / interleaved) * interleaved +
                               ((row % reordered_elements_per_thread) /
                                elements_per_thread) *
                                       instruction_shape_col +
                               ((row % interleaved) /
                                reordered_elements_per_thread) *
                                       elements_per_thread +
                               (row % elements_per_thread)
                     : row;

    uint32_t dst_offset =
            (col * OC + row) * interleaved + residue * elements_per_lane;

    *(reinterpret_cast<int4*>(dst + dst_offset * size_bits / 8)) =
            *(reinterpret_cast<const int4*>(src + src_offset * size_bits / 8));
}

template <uint32_t size_bits, uint32_t interleaved>
__global__ void reorder_ncxhwx_imma_filter_kernel(
        int8_t* __restrict__ dst_filter, const int8_t* __restrict__ src_filter,
        uint32_t OC, uint32_t IC, uint32_t FH, uint32_t FW, bool trans_oc) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    const uint32_t size = OC * IC * FH * FW / elements_per_lane;
    uint32_t lane = threadIdx.x + blockIdx.x * blockDim.x;
    if (lane < size) {
        reorder_ncxhwx_imma_filter_func<size_bits, interleaved>(
                dst_filter, src_filter, OC, IC, FH, FW, lane, trans_oc);
    }
}

template <uint32_t size_bits, uint32_t alignbits, uint32_t interleaved>
__device__ __forceinline__ void reorder_nhwc_imma_filter_func(
        int8_t* dst, const int8_t* src, uint32_t OC, uint32_t IC, uint32_t FH,
        uint32_t FW, uint32_t lane, bool trans_oc) {
    static constexpr uint32_t elements_per_access = alignbits / size_bits;
    static constexpr uint32_t instruction_shape_col = 8;
    // 4 threads per Quad
    static constexpr uint32_t elements_per_thread = instruction_shape_col / 4;
    // 4 threads per Quad
    static constexpr uint32_t reordered_elements_per_thread = interleaved / 4;
    uint32_t ICx = IC / elements_per_access;
    uint32_t k = lane / (ICx * FH * FW);
    uint32_t cxrs = lane - k * ICx * FH * FW;
    uint32_t rs = cxrs / ICx;
    uint32_t cx = cxrs - rs * ICx;
    // transpose nhwc to ncxhwx
    uint32_t src_offset = lane * elements_per_access;
    // reorder k
    k = (trans_oc)
                ? (k / interleaved) * interleaved +
                          ((k % reordered_elements_per_thread) /
                           elements_per_thread) *
                                  instruction_shape_col +
                          ((k % interleaved) / reordered_elements_per_thread) *
                                  elements_per_thread +
                          (k % elements_per_thread)
                : k;
    uint32_t dst_offset =
            (k * ICx * FH * FW + cx * FH * FW + rs) * elements_per_access;

    if (alignbits == 32) {
        *(reinterpret_cast<int*>(dst + dst_offset * size_bits / 8)) = *(
                reinterpret_cast<const int*>(src + src_offset * size_bits / 8));
    } else if (alignbits == 64) {
        *(reinterpret_cast<int2*>(dst + dst_offset * size_bits / 8)) =
                *(reinterpret_cast<const int2*>(src +
                                                src_offset * size_bits / 8));
    } else {
        *(reinterpret_cast<int4*>(dst + dst_offset * size_bits / 8)) =
                *(reinterpret_cast<const int4*>(src +
                                                src_offset * size_bits / 8));
    }
}

template <uint32_t size_bits, uint32_t alignbits, uint32_t interleaved>
__global__ void reorder_nhwc_imma_filter_kernel(
        int8_t* __restrict__ dst_filter, const int8_t* __restrict__ src_filter,
        uint32_t OC, uint32_t IC, uint32_t FH, uint32_t FW, bool trans_oc) {
    static constexpr uint32_t elements_per_access = alignbits / size_bits;
    const uint32_t size = OC * IC * FH * FW / elements_per_access;
    uint32_t lane = threadIdx.x + blockIdx.x * blockDim.x;
    if (lane < size) {
        reorder_nhwc_imma_filter_func<size_bits, alignbits, interleaved>(
                dst_filter, src_filter, OC, IC, FH, FW, lane, trans_oc);
    }
}
}  // namespace

template <uint32_t size_bits, uint32_t interleaved>
void megdnn::cuda::cutlass_wrapper::reorder_ncxhwx_imma_filter(
        int8_t* dst_filter, const int8_t* src_filter, uint32_t OC, uint32_t IC,
        uint32_t FH, uint32_t FW, bool trans_oc, cudaStream_t stream) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    uint32_t nr_threads =
            query_blocksize_for_kernel(reinterpret_cast<const void*>(
                    reorder_ncxhwx_imma_filter_kernel<size_bits, interleaved>));
    uint32_t vthreads = DIVUP(OC * IC * FH * FW, elements_per_lane);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    reorder_ncxhwx_imma_filter_kernel<size_bits, interleaved>
            <<<nr_blocks, nr_threads, 0, stream>>>(dst_filter, src_filter, OC,
                                                   IC, FH, FW, trans_oc);
    after_kernel_launch();
}

template <uint32_t size_bits, uint32_t alignbits>
void megdnn::cuda::cutlass_wrapper::reorder_nhwc_imma_filter(
        int8_t* dst_filter, const int8_t* src_filter, uint32_t OC, uint32_t IC,
        uint32_t FH, uint32_t FW, bool trans_oc, uint32_t oc_interleaved,
        cudaStream_t stream) {
    static constexpr uint32_t elements_per_access = alignbits / size_bits;
    uint32_t nr_threads =
            query_blocksize_for_kernel(reinterpret_cast<const void*>(
                    reorder_nhwc_imma_filter_kernel<size_bits, alignbits, 32>));
    uint32_t vthreads = DIVUP(OC * IC * FH * FW, elements_per_access);
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    if (oc_interleaved == 32) {
        reorder_nhwc_imma_filter_kernel<size_bits, alignbits, 32>
                <<<nr_blocks, nr_threads, 0, stream>>>(
                        dst_filter, src_filter, OC, IC, FH, FW, trans_oc);
    } else {
        reorder_nhwc_imma_filter_kernel<size_bits, alignbits, 64>
                <<<nr_blocks, nr_threads, 0, stream>>>(
                        dst_filter, src_filter, OC, IC, FH, FW, trans_oc);
    }
    after_kernel_launch();
}

#define INST(_size_bits, _interleaved)                                       \
    template void megdnn::cuda::cutlass_wrapper::reorder_ncxhwx_imma_filter< \
            _size_bits, _interleaved>(int8_t * dst_filter,                   \
                                      const int8_t* src_filter, uint32_t OC, \
                                      uint32_t IC, uint32_t FH, uint32_t FW, \
                                      bool trans_oc, cudaStream_t stream);

INST(8, 32)
INST(4, 64)
#undef INST

#define INST(_size_bits, _alignbits)                                       \
    template void megdnn::cuda::cutlass_wrapper::reorder_nhwc_imma_filter< \
            _size_bits, _alignbits>(                                       \
            int8_t * dst_filter, const int8_t* src_filter, uint32_t OC,    \
            uint32_t IC, uint32_t FH, uint32_t FW, bool trans_oc,          \
            uint32_t oc_interleaved, cudaStream_t stream);
INST(4, 32)
INST(4, 64)
INST(4, 128)
#undef INST

// vim: syntax=cuda.doxygen
