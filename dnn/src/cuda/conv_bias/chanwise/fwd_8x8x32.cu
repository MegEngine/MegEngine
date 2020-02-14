/**
 * \file dnn/src/cuda/conv_bias/chanwise/fwd_8x8x32.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/conv_bias/chanwise/kern.cuh"

#include <cassert>
#include <cstdio>

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;
using namespace chanwise;

namespace {

__host__ __device__ void get_receptive_field_size(uint32_t OH, uint32_t OW,
                                                  uint32_t FH, uint32_t FW,
                                                  uint32_t SH, uint32_t SW,
                                                  uint32_t DH, uint32_t DW,
                                                  uint32_t* RH, uint32_t* RW) {
    // DFH = dilationd FH, DFW = dilationd FW
    // RH = receptive field height, RW = receptive field width
    uint32_t DFH = (FH - 1) * DH + 1, DFW = (FW - 1) * DW + 1;
    *RH = ((OH - 1) * SH + 1) + DFH - 1;
    *RW = ((OW - 1) * SW + 1) + DFW - 1;
}

// 32x4x4 threads
// assume that C must be multiples of 4
// F == 0: FH/FW should be retrieved from param
// F != 0: FH/FW should use F
template <uint32_t F>
__global__ void kern(int32_t* dst, const int8_t* src, const int8_t* flt,
                     Param param) {
    // each block would process 128 channels at every 4x4 spatial area.
    uint32_t C = param.src_chl, IH = param.src_h, IW = param.src_w,
             OH = param.out_h, OW = param.out_w, FH = F == 0 ? param.flt_h : F,
             FW = F == 0 ? param.flt_w : F, PH = param.pad_h, PW = param.pad_w,
             SH = param.stride_h, SW = param.stride_w, DH = param.dilation_h,
             DW = param.dilation_w;

    const uint32_t* src_32 = reinterpret_cast<const uint32_t*>(src);
    const uint32_t* flt_32 = reinterpret_cast<const uint32_t*>(flt);
    uint32_t bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
    uint32_t c_beg = blockIdx.x * 128, c_end = min((blockIdx.x + 1) * 128, C),
             c_cur = c_beg + threadIdx.x * 4;
    uint32_t tidx = threadIdx.x, tidy = threadIdx.y, tidz = threadIdx.z,
             tid = (tidx << 0) | (tidy << 5) | (tidz << 7),
             tid_stride = 32 * 4 * 4, tidyz = (tidy << 0) | (tidz << 2),
             tidyz_stride = 4 * 4;
    uint32_t oh = bidz * 4 + tidz, ow = bidy * 4 + tidy;
    uint32_t C_32 = C >> 2;
    // calculate receptive field of 4x4 output pixels
    uint32_t RH, RW;
    get_receptive_field_size(4, 4, FH, FW, SH, SW, DH, DW, &RH, &RW);

    extern __shared__ int8_t shared[];

    int8_t* flt_shared_tmp = static_cast<int8_t*>(static_cast<void*>(shared));
    uint32_t* flt_shared_tmp_32 = reinterpret_cast<uint32_t*>(flt_shared_tmp);

    int8_t* flt_shared = static_cast<int8_t*>(
            static_cast<void*>(shared + 128 * FH * FW * sizeof(int8_t)));
    uint32_t* flt_shared_32 = reinterpret_cast<uint32_t*>(flt_shared);

    int8_t* src_shared = static_cast<int8_t*>(
            static_cast<void*>(shared + 128 * FH * FW * sizeof(int8_t) +
                               128 * FH * FW * sizeof(int8_t)));
    uint32_t* src_shared_32 = reinterpret_cast<uint32_t*>(src_shared);

    int32_t* dst_shared = static_cast<int32_t*>(static_cast<void*>(
            shared + 128 * FH * FW * sizeof(int8_t) +
            128 * FH * FW * sizeof(int8_t) + 128 * RH * RW * sizeof(int8_t)));

    // read original filter to shared memory
    // *_int8 vars must be multiples of 4 here.
    uint32_t flt_offset = c_beg * FH * FW;
    uint32_t flt_offset_32 = flt_offset >> 2;
    uint32_t flt_amount = (c_end - c_beg) * FH * FW;
    uint32_t flt_amount_32 = flt_amount >> 2;
    for (uint32_t id = tid; id < flt_amount_32; id += tid_stride) {
        flt_shared_tmp_32[id] = flt_32[flt_offset_32 + id];
    }
    __syncthreads();
    // transpose filter: (flt_amount, FH*FW) -> (FH*FW, 128)
    // typical example: (128, 9) -> (9, 128)
    for (uint32_t idyz = tidyz; idyz < FH * FW; idyz += tidyz_stride)
        for (uint32_t idx = tidx; idx < 128; idx += 32) {
            uint32_t from_idx = idx * FH * FW + idyz;
            uint32_t to_idx = idx + idyz * 128;
            if (from_idx < flt_amount) {
                flt_shared[to_idx] = flt_shared_tmp[from_idx];
            } else {
                flt_shared[to_idx] = 0;
            }
        }
    // no need to sync here
    // __syncthreads();
    // read (RH, RW, 128) src from global to shared
    for (uint32_t rh = tidz; rh < RH; rh += 4)
        for (uint32_t rw = tidy; rw < RW; rw += 4) {
            uint32_t ih = bidz * 4 * SH + rh - PH;
            uint32_t iw = bidy * 4 * SW + rw - PW;
            uint32_t to_idx = (rh * RW + rw) * 32 + tidx;
            uint32_t c_32 = bidx * 32 + tidx;
            uint32_t from_idx = (ih * IW + iw) * C_32 + c_32;
            if (ih < IH && iw < IW && c_32 < C_32) {
                src_shared_32[to_idx] = src_32[from_idx];
            } else {
                src_shared_32[to_idx] = 0;
            }
        }
    __syncthreads();
    // do convolution
    if (c_cur < c_end && oh < OH && ow < OW) {
        int32_t dst0 = 0, dst1 = 0, dst2 = 0, dst3 = 0;
#pragma unroll
        for (uint32_t fh = 0; fh < FH; ++fh)
#pragma unroll
            for (uint32_t fw = 0; fw < FW; ++fw) {
                uint32_t rh = tidz * SH + fh * DH, rw = tidy * SW + fw * DW;
                uint32_t sval_32 = src_shared_32[(rh * RW + rw) * 32 + tidx];
                int32_t sval0 = int8_t((sval_32 >> 0) & 255),
                        sval1 = int8_t((sval_32 >> 8) & 255),
                        sval2 = int8_t((sval_32 >> 16) & 255),
                        sval3 = int8_t((sval_32 >> 24) & 255);
                uint32_t fval_32 = flt_shared_32[(fh * FW + fw) * 32 + tidx];
                int32_t fval0 = int8_t((fval_32 >> 0) & 255),
                        fval1 = int8_t((fval_32 >> 8) & 255),
                        fval2 = int8_t((fval_32 >> 16) & 255),
                        fval3 = int8_t((fval_32 >> 24) & 255);
                dst0 += sval0 * fval0;
                dst1 += sval1 * fval1;
                dst2 += sval2 * fval2;
                dst3 += sval3 * fval3;
            }
        dst_shared[tidyz * 129 + tidx * 4 + 0] = dst0;
        dst_shared[tidyz * 129 + tidx * 4 + 1] = dst1;
        dst_shared[tidyz * 129 + tidx * 4 + 2] = dst2;
        dst_shared[tidyz * 129 + tidx * 4 + 3] = dst3;
    }
    __syncthreads();
    if (oh < OH && ow < OW) {
#pragma unroll
        for (uint32_t k = 0; k < 4; ++k) {
            uint32_t c = c_beg + tidx + k * 32;
            if (c < c_end) {
                dst[(oh * OW + ow) * C + c] =
                        dst_shared[tidyz * 129 + tidx + k * 32];
            }
        }
    }
}

}  // anonymous namespace

void megdnn::cuda::conv_bias::chanwise::run_fwd_8x8x32(int32_t* dst,
                                                       const int8_t* src,
                                                       const int8_t* flt,
                                                       const Param& param,
                                                       cudaStream_t stream) {
    uint32_t N = param.batch, C = param.src_chl, IH = param.src_h,
             IW = param.src_w, OH = param.out_h, OW = param.out_w,
             FH = param.flt_h, FW = param.flt_w, SH = param.stride_h,
             SW = param.stride_w, DH = param.dilation_h, DW = param.dilation_w;

    dim3 threads(32, 4, 4);
    dim3 blocks(DIVUP(C, 128), DIVUP(OW, 4), DIVUP(OH, 4));

    // shared mem size: filter*2 + src + dst
    // filter
    uint32_t filter_shared_mem_size = 128 * FH * FW * sizeof(int8_t);
    // src
    uint32_t RH, RW;
    get_receptive_field_size(4, 4, FH, FW, SH, SW, DH, DW, &RH, &RW);
    uint32_t src_shared_mem_size = 128 * RH * RW * sizeof(int8_t);
    // dst
    // use 129 instead of 128 to avoid shared memory bank conflict
    uint32_t dst_shared_mem_size = 129 * 4 * 4 * sizeof(int32_t);

    uint32_t shared_mem_size = 2 * filter_shared_mem_size +
                               src_shared_mem_size + dst_shared_mem_size;

    void (*kptr)(int32_t*, const int8_t*, const int8_t*, Param) = kern<0>;
    if (FH == 1 && FW == 1)
        kptr = kern<1>;
    if (FH == 3 && FW == 3)
        kptr = kern<3>;
    if (FH == 5 && FW == 5)
        kptr = kern<5>;

    for (uint32_t n = 0; n < N; ++n) {
        int32_t* dptr = dst + n * C * OH * OW;
        const int8_t* sptr = src + n * C * IH * IW;
        const int8_t* fptr = flt;
        kptr<<<blocks, threads, shared_mem_size, stream>>>(dptr, sptr, fptr,
                                                           param);
    }
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen
