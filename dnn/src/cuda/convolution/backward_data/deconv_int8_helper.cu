/**
 * \file src/cuda/convolution/backward_data/deconv_int8_helper.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/convolution/backward_data/deconv_int8_helper.cuh"
#include "src/cuda/transpose_utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace deconv;

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

namespace {

__global__ void reorder_filter_nc4hw4_to_n4hwc4_kernel(
        int8_t* __restrict__ dst, const int8_t* __restrict__ src, uint32_t OC,
        uint32_t IC, uint32_t FHFW) {
    const int32_t ocb = blockIdx.z;
    const int32_t icb = blockIdx.y * BLOCKSIZE_X + threadIdx.y;
    const int32_t fhfw = blockIdx.x * BLOCKSIZE_Y + threadIdx.x;

    if (fhfw < FHFW && icb < IC / 4) {
        array_wrapper<int, 4> src_value;
        int dst_value[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            src_value[i] = *reinterpret_cast<const int*>(
                    src + (ocb * 4 + i) * IC * FHFW + (icb * FHFW + fhfw) * 4);
        }

        // transpose 4x4
        auto trans = transpose_int8_interleavedx4<4, int>();
        trans(src_value, dst_value);

#pragma unroll
        for (int i = 0; i < 4; i++) {
            *reinterpret_cast<int*>(
                    dst + (ocb * FHFW * IC + fhfw * IC + icb * 4 + i) * 4) =
                    dst_value[i];
        }
    }
}

template <uint32_t interleaved, typename vec_type>
__global__ void reorder_filter_nhwc_to_cnxhwx_kernel(
        int8_t* __restrict__ dst, const int8_t* __restrict__ src, uint32_t OC,
        uint32_t IC, uint32_t FHFW) {
    uint32_t lane = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t ocb = lane / (FHFW * IC / 4);
    const int32_t fhfw_icb = lane % (FHFW * IC / 4);
    const int32_t fhfw = fhfw_icb / (IC / 4);
    const int32_t icb = fhfw_icb % (IC / 4);

    if (ocb < OC / interleaved && fhfw < FHFW) {
        array_wrapper<int, interleaved> src_value;
        vec_type dst_value[4];

#pragma unroll
        for (int i = 0; i < interleaved; i++) {
            src_value[i] = *reinterpret_cast<const int*>(
                    src + (ocb * interleaved + i) * FHFW * IC + fhfw * IC +
                    icb * 4);
        }

        auto trans = transpose_int8_interleavedx4<interleaved, vec_type>();
        trans(src_value, dst_value);

#pragma unroll
        for (int i = 0; i < 4; i++) {
            *reinterpret_cast<vec_type*>(dst + (icb * 4 + i) * FHFW * OC +
                                         (ocb * FHFW + fhfw) * interleaved) =
                    dst_value[i];
        }
    }
}

}  // namespace

void megdnn::cuda::deconv::reorder_filter_nc4hw4_to_n4hwc4(
        int8_t* dst, const int8_t* src, uint32_t OC, uint32_t IC, uint32_t FH,
        uint32_t FW, cudaStream_t stream) {
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 blocks(DIVUP(FH * FW, BLOCKSIZE_X), DIVUP(IC / 4, BLOCKSIZE_Y),
                OC / 4);

    reorder_filter_nc4hw4_to_n4hwc4_kernel<<<blocks, threads, 0, stream>>>(
            dst, src, OC, IC, FH * FW);
    after_kernel_launch();
}

void megdnn::cuda::deconv::reorder_filter_nhwc_to_cnxhwx(
        int8_t* dst, const int8_t* src, uint32_t OC, uint32_t IC, uint32_t FH,
        uint32_t FW, uint32_t interleaved, cudaStream_t stream) {
    int32_t vthreads = OC / interleaved * IC / 4 * FH * FW;
    int32_t nr_threads = std::min(256, vthreads);
    int32_t nr_blocks = DIVUP(vthreads, nr_threads);

    if (interleaved == 4) {
        reorder_filter_nhwc_to_cnxhwx_kernel<4, int>
                <<<nr_blocks, nr_threads, 0, stream>>>(dst, src, OC, IC,
                                                       FH * FW);
    } else if (interleaved == 8) {
        reorder_filter_nhwc_to_cnxhwx_kernel<8, int2>
                <<<nr_blocks, nr_threads, 0, stream>>>(dst, src, OC, IC,
                                                       FH * FW);
    } else {
        reorder_filter_nhwc_to_cnxhwx_kernel<16, int4>
                <<<nr_blocks, nr_threads, 0, stream>>>(dst, src, OC, IC,
                                                       FH * FW);
    }
    after_kernel_launch();
}

// vim: syntax=cuda.doxygen
