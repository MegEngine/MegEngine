/**
 * \file dnn/src/cuda/conv_bias/matmul/im2col_nhwc_int8.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/conv_bias/matmul/im2col_nhwc_int8.cuh"

#include "src/cuda/utils.cuh"

namespace {

template <bool flip>
__global__ void im2col_kern(const int8_t* __restrict src,
                            int8_t* __restrict unrolled, uint32_t N,
                            uint32_t IH, uint32_t IW, uint32_t IC, uint32_t IWS,
                            uint32_t OH, uint32_t OW, uint32_t OC, uint32_t OWS,
                            uint32_t FH, uint32_t FW, uint32_t PH, uint32_t PW,
                            uint32_t SH, uint32_t SW, uint32_t DH, uint32_t DW,
                            uint32_t LD) {
    uint32_t ic = blockIdx.x * 32 + threadIdx.x;
    uint32_t ow = blockIdx.y * 4 + threadIdx.y;
    uint32_t oh = blockIdx.z * 4 + threadIdx.z;
    uint32_t offset = (oh * OW + ow) * LD + ic;
    if (ic < IC && ow < OW && oh < OH) {
        for (uint32_t fh = 0; fh < FH; ++fh) {
            for (size_t fw = 0; fw < FW; ++fw) {
                uint32_t ih = -PH + oh * SH + (flip ? FH - fh - 1 : fh) * DH;
                uint32_t iw = -PW + ow * SW + (flip ? FW - fw - 1 : fw) * DW;
                uint32_t i = offset + (fh * FW + fw) * IC;
                if (ih < IH && iw < IW) {
                    unrolled[i] = src[(ih * IW + iw) * IWS + ic];
                } else {
                    unrolled[i] = 0;
                }
            }
        }
    }
}

}  // anonymous namespace

void megdnn::cuda::im2col_nhwc_int8(const int8_t* src, int8_t* unrolled,
                                    uint32_t N, uint32_t IH, uint32_t IW,
                                    uint32_t IC, uint32_t IWS, uint32_t OH,
                                    uint32_t OW, uint32_t OC, uint32_t OWS,
                                    uint32_t FH, uint32_t FW, uint32_t PH,
                                    uint32_t PW, uint32_t SH, uint32_t SW,
                                    uint32_t DH, uint32_t DW, uint32_t LD,
                                    bool flip, cudaStream_t stream) {
    dim3 nthreads = dim3(32, 4, 4);
    dim3 nblocks = dim3(DIVUP(IC, 32), DIVUP(OW, 4), DIVUP(OH, 4));
    void (*kern_ptr)(const int8_t* __restrict src, int8_t* __restrict unrolled,
                     uint32_t N, uint32_t IH, uint32_t IW, uint32_t IC,
                     uint32_t IWS, uint32_t OH, uint32_t OW, uint32_t OC,
                     uint32_t OWS, uint32_t FH, uint32_t FW, uint32_t PH,
                     uint32_t PW, uint32_t SH, uint32_t SW, uint32_t DH,
                     uint32_t DW, uint32_t LD);
    if (flip) {
        kern_ptr = im2col_kern<true>;
    } else {
        kern_ptr = im2col_kern<false>;
    }
    for (size_t n = 0; n < N; ++n) {
        kern_ptr<<<nblocks, nthreads, 0, stream>>>(
                src + n * IH * IW * IWS, unrolled + n * OH * OW * LD, N, IH, IW,
                IC, IWS, OH, OW, OC, OWS, FH, FW, PH, PW, SH, SW, DH, DW, LD);
    }
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen
