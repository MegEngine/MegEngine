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

using namespace megdnn;
using namespace cuda;
using namespace deconv;

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

namespace {

//
__global__ void reorder_filter_nc4hw4_to_n4hwc4_kernel(
        int8_t* __restrict__ dst, const int8_t* __restrict__ src, uint32_t OC,
        uint32_t IC, uint32_t FHFW) {
    const int32_t ocb = blockIdx.z;
    const int32_t icb = blockIdx.y * BLOCKSIZE_X + threadIdx.y;
    const int32_t fhfw = blockIdx.x * BLOCKSIZE_Y + threadIdx.x;

    if (fhfw < FHFW && icb < IC / 4) {
        int src0 = *reinterpret_cast<const int*>(
                src + (ocb * 4 + 0) * IC * FHFW + (icb * FHFW + fhfw) * 4);
        int src1 = *reinterpret_cast<const int*>(
                src + (ocb * 4 + 1) * IC * FHFW + (icb * FHFW + fhfw) * 4);
        int src2 = *reinterpret_cast<const int*>(
                src + (ocb * 4 + 2) * IC * FHFW + (icb * FHFW + fhfw) * 4);
        int src3 = *reinterpret_cast<const int*>(
                src + (ocb * 4 + 3) * IC * FHFW + (icb * FHFW + fhfw) * 4);
        // transpose 4x4
        int dst01_lo = __byte_perm(src0, src1, 0x5140);
        int dst01_hi = __byte_perm(src0, src1, 0x7362);
        int dst23_lo = __byte_perm(src2, src3, 0x5140);
        int dst23_hi = __byte_perm(src2, src3, 0x7362);
        int dst0 = __byte_perm(dst01_lo, dst23_lo, 0x5410);
        int dst1 = __byte_perm(dst01_lo, dst23_lo, 0x7632);
        int dst2 = __byte_perm(dst01_hi, dst23_hi, 0x5410);
        int dst3 = __byte_perm(dst01_hi, dst23_hi, 0x7632);

        *reinterpret_cast<int*>(
                dst + (ocb * FHFW * IC + fhfw * IC + icb * 4 + 0) * 4) = dst0;
        *reinterpret_cast<int*>(
                dst + (ocb * FHFW * IC + fhfw * IC + icb * 4 + 1) * 4) = dst1;
        *reinterpret_cast<int*>(
                dst + (ocb * FHFW * IC + fhfw * IC + icb * 4 + 2) * 4) = dst2;
        *reinterpret_cast<int*>(
                dst + (ocb * FHFW * IC + fhfw * IC + icb * 4 + 3) * 4) = dst3;
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

// vim: syntax=cuda.doxygen
