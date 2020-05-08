/**
 * \file dnn/src/cuda/group_local/forward/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/cuda/group_local/forward/kern.cuh"

#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;

namespace {

constexpr size_t NB = 4, ICB = 4;

// src layout is (N, G, IC, IH, IW)
// filter layout is (G, OH, OW, IC, FH, FW, OC)
// dst layout is (N, G, OC, OH, OW)
// NR_THREADS is 256
// gridDim.z is G
// gridDim.y is OC*OH*OW/NR_THREADS
// gridDim.x is N/NB
// blockDim.x is NR_THREADS

// INs and ONs are the stride on the src/dst batch size dim
// IC and OC are nr. channels per group

// Each thread tackles with NB (actually NB_cur if non-multiple-of-NB N is
// considered). Let oid = blockIdx.y*NR_THREADS + threadIdx.x (global thread ID
// along block axis y), and we flatten (OC, OH, OW) into one dimension, then
// each thread calculates the answer at dst position (n, blockIdx.z, oid), where
// n ranges from blockDim.x*NB + 0 to blockDim.x*NB + (NB-1). IC is processed at
// stride of ICB. On entrance of each iteration of the loop, NB * ICB spatial
// src planes are loaded into shared memory (presumably src spatial size is
// small).
template <uint32_t NB, uint32_t ICB, bool is_xcorr>
__global__ void forward_kernel(const float* __restrict__ src,
                               const float* __restrict__ filter,
                               float* __restrict__ dst, uint32_t N, uint32_t IC,
                               uint32_t IH, uint32_t IW, uint32_t OC,
                               uint32_t OH, uint32_t OW, uint32_t FH,
                               uint32_t FW, uint32_t INs, uint32_t ONs,
                               uint32_t PH, uint32_t PW, uint32_t SH,
                               uint32_t SW) {
    // NB * ICB * sizeof(float) * IH * IW
    extern __shared__ float shared_mem[];
    float* src_cache = shared_mem;
    uint32_t tid = threadIdx.x;
    uint32_t tstride = blockDim.x;
    uint32_t oid = tid + blockIdx.y * tstride;
    src += blockIdx.x * NB * INs + blockIdx.z * IC * IH * IW;
    dst += blockIdx.x * NB * ONs + blockIdx.z * OC * OH * OW;
    filter += blockIdx.z * OH * OW * IC * FH * FW * OC;
    uint32_t op = oid / OC;
    uint32_t oc = oid % OC;
    uint32_t oh = op / OW;
    uint32_t ow = op % OW;
    float dst_reg[NB];
    for (uint32_t nb = 0; nb < NB; ++nb)
        dst_reg[nb] = 0.0f;
    uint32_t NB_cur = min(N - blockIdx.x * NB, NB);
    for (uint32_t ic = 0; ic < IC; ic += ICB) {
        // read ICB-channel src
        // (NB, ICB, IHs, IWs)
        uint32_t ICB_cur = min(ICB, IC - ic);
        for (uint32_t i = tid; i < NB_cur * ICB * IH * IW; i += tstride) {
            uint32_t ip = i % (IH * IW);
            uint32_t icb = i / (IH * IW) % ICB;
            uint32_t nb = i / (IH * IW) / ICB;
            src_cache[i] =
                    (icb < ICB_cur) *
                    src[nb * INs + min(IC - 1, (ic + icb)) * IH * IW + ip];
        }
        __syncthreads();
        if (oid < OC * OH * OW)
            for (uint32_t fh = 0; fh < FH; ++fh) {
                uint32_t ih;
                if (is_xcorr)
                    ih = oh * SH + fh - PH;
                else
                    ih = oh * SH + (FH - fh - 1) - PH;
                if (ih < IH)
                    for (uint32_t fw = 0; fw < FW; ++fw) {
                        uint32_t iw;
                        if (is_xcorr)
                            iw = ow * SW + fw - PW;
                        else
                            iw = ow * SW + (FW - fw - 1) - PW;
                        if (iw < IW)
                            for (uint32_t icb = 0; icb < ICB_cur; ++icb) {
                                uint32_t fid = op * IC * FH * FW * OC +
                                               (ic + icb) * FH * FW * OC +
                                               fh * FW * OC + fw * OC + oc;
                                float fval = filter[fid];
                                float src_reg[NB];
#pragma unroll
                                for (uint32_t nb = 0; nb < NB; ++nb) {
                                    src_reg[nb] = src_cache[nb * ICB * IH * IW +
                                                            icb * IH * IW +
                                                            ih * IW + iw];
                                }
#pragma unroll
                                for (uint32_t nb = 0; nb < NB; ++nb) {
                                    dst_reg[nb] += src_reg[nb] * fval;
                                }
                            }
                    }
            }
        __syncthreads();
    }
    if (oid < OC * OH * OW) {
        for (uint32_t nb = 0; nb < NB_cur; ++nb) {
            dst[nb * ONs + oc * OH * OW + op] = dst_reg[nb];
        }
    }
}

}

void group_local::exec(const float* src, const float* filter, float* dst,
                       float* wptr, uint32_t N, uint32_t IC, uint32_t IH,
                       uint32_t IW, uint32_t OC, uint32_t OH, uint32_t OW,
                       uint32_t FH, uint32_t FW, uint32_t G, uint32_t PH,
                       uint32_t PW, uint32_t SH, uint32_t SW,
                       cudaStream_t stream) {
    MEGDNN_MARK_USED_VAR(wptr);
    size_t threads = 256;
    dim3 blocks = dim3(DIVUP(N, NB), DIVUP(OC * OH * OW, threads), G);
    uint32_t INs = G * IC * IH * IW, ONs = G * OC * OH * OW;
    forward_kernel<NB, ICB, true>
            <<<blocks, threads, NB * ICB * sizeof(float) * IH * IW, stream>>>(
                    src, filter, dst, N, IC, IH, IW, OC, OH, OW, FH, FW, INs,
                    ONs, PH, PW, SH, SW);
    after_kernel_launch();
}

size_t group_local::get_share_mem_in_bytes(uint32_t IH, uint32_t IW) {
    return NB * ICB * sizeof(float) * IH * IW;
}
