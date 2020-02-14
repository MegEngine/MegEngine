/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma/wmma_conv_integer_u4_fhxfw.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <stdio.h>
#include "src/cuda/utils.cuh"
#include "wmma_conv_integer_u4.cuh"

#if __CUDA_ARCH__ >= 730
using namespace nvcuda;
using namespace wmma::experimental::precision;
#endif

using namespace megdnn;
using namespace cuda;
using namespace wmma_conv_integer_subbyte;

namespace wmma_conv_integer_subbyte_fhxfw {

template <int WARPS_W_, int WARPS_OC_, int OUT_CHANNELS_PER_WARP_,
          int OH_PER_WARP_, int IC_UNROLL_SIZE_>
struct BlockConfig {
    static int const WARPS_W = WARPS_W_;
    static int const WARPS_OC = WARPS_OC_;
    static int const OUT_CHANNELS_PER_WARP = OUT_CHANNELS_PER_WARP_;
    static int const OH_PER_WARP = OH_PER_WARP_;
    static int const IC_UNROLL_SIZE = IC_UNROLL_SIZE_;
    static int const IC_BLKS = IC_BLK * IC_UNROLL_SIZE;
    static int const WARPS_PER_BLOCK = WARPS_W * WARPS_OC;
};

template <typename ConvConfig, typename BlockConfig>
struct DataCount {
    static int const LANE_SIZE =
            BlockConfig::WARPS_W * WMMA_M * ConvConfig::SW + ConvConfig::FW - 1;
    static int const LANES_PER_SLICE =
            BlockConfig::OH_PER_WARP * ConvConfig::SH + ConvConfig::FH - 1;
    static int const LANES_PER_BLOCK =
            LANES_PER_SLICE * IC_BLK * BlockConfig::IC_UNROLL_SIZE;
    static int const LANES_PER_WARP =
            (LANES_PER_BLOCK + BlockConfig::WARPS_PER_BLOCK - 1) /
            BlockConfig::WARPS_PER_BLOCK;
    static int const SMEM_SKEW = (BlockConfig::IC_UNROLL_SIZE % 2 == 0) * SKEW;
    static int const SMEM_DATA_COL =
            (IC_BLK * BlockConfig::IC_UNROLL_SIZE * 8 + SMEM_SKEW) / 2;
    static int const SMEM_DATA_STRIDE = SMEM_DATA_COL * 2;
    static int const SMEM_DATA_ROW = LANE_SIZE * LANES_PER_SLICE;
};

template <typename ConvConfig, typename BlockConfig>
struct FilterCount {
    static int const OUT_CHANNELS_PER_BLOCK =
            WMMA_M * BlockConfig::WARPS_OC * BlockConfig::OUT_CHANNELS_PER_WARP;
    static int const SMEM_FILTER_ROW = OUT_CHANNELS_PER_BLOCK;
    static int const SMEM_SKEW =
            ((ConvConfig::FH * ConvConfig::FW * BlockConfig::IC_UNROLL_SIZE) %
                     2 ==
             0) *
            SKEW;
    static int const SMEM_FILTER_COL =
            (BlockConfig::IC_BLKS * ConvConfig::FH * ConvConfig::FW * 8 +
             SMEM_SKEW) /
            2;
    static int const SMEM_FILTER_STRIDE = SMEM_FILTER_COL * 2;
    static int const REG_FILTER_ROW =
            (SMEM_FILTER_ROW + BlockConfig::WARPS_PER_BLOCK - 1) /
            BlockConfig::WARPS_PER_BLOCK;
    static int const REG_FILTER_COL =
            (BlockConfig::IC_BLKS * ConvConfig::FH * ConvConfig::FW +
             WARP_SIZE - 1) /
            WARP_SIZE;
};

#if __CUDA_ARCH__ >= 730
template <typename ConvConfig_, typename BlockConfig_>
struct ConvDataGlobal2ShareMemVisitor {
    typedef int32_t copy_t;
    uint8_t* smem;
    const uint8_t* g_ptr;

    int ci_stride, hi_stride;
    int b_ih, b_iw;
    int IH, IW;
    copy_t zero;
    const int warp_x = threadIdx.x / WARP_SIZE;
    const int warp_y = threadIdx.y;
    const int tid_in_warp = threadIdx.x % WARP_SIZE;
    const int warp_id = (warp_y * BlockConfig_::WARPS_W + warp_x);

    copy_t reg_cache[DataCount<ConvConfig_, BlockConfig_>::LANES_PER_WARP];

    __device__ ConvDataGlobal2ShareMemVisitor(uint8_t* smem,
                                              const uint8_t* g_ptr, int IH,
                                              int IW, int b_ih, int b_iw,
                                              copy_t zero)
            : smem{smem},
              g_ptr{g_ptr},
              b_ih{b_ih},
              b_iw{b_iw},
              IH{IH},
              IW{IW},
              zero{zero} {
        ci_stride = 8 * IH * IW;
        hi_stride = 8 * IW;
    }

    // not perfectly
    __device__ __forceinline__ void copy() {
        int col = (tid_in_warp << 3);
        // read input from global memory without boundary check
#pragma unroll
        for (int i = 0;
             i < DataCount<ConvConfig_, BlockConfig_>::LANES_PER_WARP; ++i) {
            int row = i * BlockConfig_::WARPS_PER_BLOCK + warp_id;
            int ci_idx =
                    row / DataCount<ConvConfig_, BlockConfig_>::LANES_PER_SLICE;
            int hi_idx =
                    row - ci_idx * DataCount<ConvConfig_,
                                             BlockConfig_>::LANES_PER_SLICE;
            bool bounds = ((b_iw + tid_in_warp) >= 0) &&
                          ((b_iw + tid_in_warp) < IW) &&
                          ((b_ih + hi_idx) >= 0) && ((b_ih + hi_idx) < IH);
            if (bounds) {
                copy_t val = *(copy_t*)(&g_ptr[(ci_idx * ci_stride +
                                                hi_idx * hi_stride + col) /
                                               2]);
                reg_cache[i] = val;
            } else {
                reg_cache[i] = zero;
            }
        }
    }

    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0;
             i < DataCount<ConvConfig_, BlockConfig_>::LANES_PER_WARP; ++i) {
            if (tid_in_warp < DataCount<ConvConfig_, BlockConfig_>::LANE_SIZE) {
                int row = i * BlockConfig_::WARPS_PER_BLOCK + warp_id;
                int ci_idx =
                        row /
                        DataCount<ConvConfig_, BlockConfig_>::LANES_PER_SLICE;
                int hi_idx =
                        row - ci_idx * DataCount<ConvConfig_,
                                                 BlockConfig_>::LANES_PER_SLICE;
                int y = hi_idx * DataCount<ConvConfig_,
                                           BlockConfig_>::LANE_SIZE +
                        tid_in_warp;
                int x = ci_idx * 8;
                *(copy_t*)(get_smem_ptr(y, x)) = reg_cache[i];
            }
        }
    }

    __device__ __forceinline__ uint8_t* get_smem_ptr(int y, int x) {
        return &smem[(y * DataCount<ConvConfig_,
                                    BlockConfig_>::SMEM_DATA_STRIDE +
                      x) /
                     2];
    }

    __device__ __forceinline__ void inc_stage() {
        g_ptr += BlockConfig_::IC_BLKS * ci_stride / 2;
    }
};

template <typename ConvConfig_, typename BlockConfig_>
struct ConvFilterGlobal2ShareMemVisitor {
    uint8_t* smem;
    const uint8_t* g_ptr;

    int co_stride, co_remain;
    int idx;
    const int warp_x = threadIdx.x / WARP_SIZE;
    const int warp_y = threadIdx.y;
    const int tid_in_warp = threadIdx.x % WARP_SIZE;
    const int warp_id = (warp_y * BlockConfig_::WARPS_W + warp_x);

    typedef int32_t copy_t;
    copy_t reg_cache[FilterCount<ConvConfig_, BlockConfig_>::REG_FILTER_ROW]
                    [FilterCount<ConvConfig_, BlockConfig_>::REG_FILTER_COL];

    __device__ ConvFilterGlobal2ShareMemVisitor(uint8_t* smem,
                                                const uint8_t* g_ptr,
                                                int co_stride, int co_remain,
                                                int idx)
            : smem{smem},
              g_ptr{g_ptr},
              co_stride{co_stride},
              co_remain{co_remain},
              idx{idx} {}

    __device__ __forceinline__ void copy() {
        int ci_remain =
                idx < BlockConfig_::IC_BLKS ? idx : BlockConfig_::IC_BLKS;
#pragma unroll
        for (int i = 0;
             i < FilterCount<ConvConfig_, BlockConfig_>::REG_FILTER_ROW; ++i) {
#pragma unroll
            for (int j = 0;
                 j < FilterCount<ConvConfig_, BlockConfig_>::REG_FILTER_COL;
                 ++j) {
                int y = BlockConfig_::WARPS_PER_BLOCK * i + warp_id;
                int x = WARP_SIZE * j + tid_in_warp;
                bool valid =
                        (x < ci_remain * ConvConfig_::FH * ConvConfig_::FW) &&
                        (y <
                         FilterCount<ConvConfig_,
                                     BlockConfig_>::OUT_CHANNELS_PER_BLOCK) &&
                        (y < co_remain);
                if (valid) {
                    copy_t val = *(copy_t*)(&g_ptr[y * co_stride + x * 4]);
                    reg_cache[i][j] = val;
                } else {
                    reg_cache[i][j] = 0;
                }
            }
        }
    }

    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0;
             i < FilterCount<ConvConfig_, BlockConfig_>::REG_FILTER_ROW; ++i) {
#pragma unroll
            for (int j = 0;
                 j < FilterCount<ConvConfig_, BlockConfig_>::REG_FILTER_COL;
                 ++j) {
                int y = BlockConfig_::WARPS_PER_BLOCK * i + warp_id;
                int x = WARP_SIZE * j + tid_in_warp;
                int spatial_idx = x % (ConvConfig_::FH * ConvConfig_::FW);
                int ci_blk = x / (ConvConfig_::FH * ConvConfig_::FW);
                int ci_inner_blk = (ci_blk & 0x3);
                int ci_outer_blk = (ci_blk >> 2);
                int s_x = ci_outer_blk * IC_BLK * ConvConfig_::FH *
                                  ConvConfig_::FW +
                          spatial_idx * IC_BLK + ci_inner_blk;
                bool bounds =
                        (y <
                         FilterCount<ConvConfig_,
                                     BlockConfig_>::OUT_CHANNELS_PER_BLOCK) &&
                        (x < BlockConfig_::IC_BLKS * ConvConfig_::FH *
                                     ConvConfig_::FW);
                if (bounds)
                    *(copy_t*)get_smem_ptr(y, s_x * 8) = reg_cache[i][j];
            }
        }
    }

    __device__ __forceinline__ uint8_t* get_smem_ptr(int y, int x) {
        return &smem[(y * FilterCount<ConvConfig_,
                                      BlockConfig_>::SMEM_FILTER_STRIDE +
                      x) /
                     2];
    }

    __device__ __forceinline__ void inc_stage() {
        idx -= BlockConfig_::IC_BLKS;
        g_ptr += BlockConfig_::IC_BLKS * ConvConfig_::FH * ConvConfig_::FW * 4;
    }
};

template <typename ConvConfig_, typename BlockConfig_>
__device__ inline void load_share_mem(
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::col_major>
                data_frag[BlockConfig_::OH_PER_WARP],
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::row_major>
                filter_frag[BlockConfig_::OUT_CHANNELS_PER_WARP],
        ConvDataGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>&
                gbl2smem_data_visitor,
        ConvFilterGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>&
                gbl2smem_filter_visitor,
        int data_spatial_idx, int filter_spatial_idx, int ic_blk) {
    const int warp_y = threadIdx.y;
    uint8_t* __restrict__ s_ptr_data = gbl2smem_data_visitor.get_smem_ptr(
            data_spatial_idx, ic_blk * WMMA_K);
    uint8_t* __restrict__ s_ptr_filter = gbl2smem_filter_visitor.get_smem_ptr(
            warp_y * WMMA_M,
            ic_blk * WMMA_K * ConvConfig_::FH * ConvConfig_::FW +
                    filter_spatial_idx * WMMA_K);

#pragma unroll
    for (int i = 0; i < BlockConfig_::OH_PER_WARP; ++i) {
        wmma::load_matrix_sync(
                data_frag[i],
                s_ptr_data +
                        i * DataCount<ConvConfig_, BlockConfig_>::LANE_SIZE *
                                DataCount<ConvConfig_,
                                          BlockConfig_>::SMEM_DATA_STRIDE /
                                2,
                DataCount<ConvConfig_, BlockConfig_>::SMEM_DATA_STRIDE);
    }
#pragma unroll
    for (int j = 0; j < BlockConfig_::OUT_CHANNELS_PER_WARP; ++j) {
        wmma::load_matrix_sync(
                filter_frag[j],
                s_ptr_filter +
                        j * WMMA_M * BlockConfig_::WARPS_OC *
                                FilterCount<ConvConfig_,
                                            BlockConfig_>::SMEM_FILTER_STRIDE /
                                2,
                FilterCount<ConvConfig_, BlockConfig_>::SMEM_FILTER_STRIDE);
    }
}

template <size_t OUT_CHANNELS_PER_WARP, size_t OH_PER_WARP>
__device__ inline void
calc(wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4, wmma::col_major>
             data_frag[OH_PER_WARP],
     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4, wmma::row_major>
             filter_frag[OUT_CHANNELS_PER_WARP],
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>
             acc_frag[OUT_CHANNELS_PER_WARP][OH_PER_WARP]) {
#pragma unroll
    for (int i = 0; i < OUT_CHANNELS_PER_WARP; ++i) {
#pragma unroll
        for (int j = 0; j < OH_PER_WARP; ++j) {
            wmma::mma_sync(acc_frag[i][j], filter_frag[i], data_frag[j],
                           acc_frag[i][j]);
        }
    }
}

template <bool last_slice, typename ConvConfig_, typename BlockConfig_>
__device__ void consume_slice(
        ConvDataGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>&
                gbl2smem_data_visitor,
        ConvFilterGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>&
                gbl2smem_filter_visitor,
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::col_major>
                data_frag[2][BlockConfig_::OH_PER_WARP],
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::row_major>
                filter_frag[2][BlockConfig_::OUT_CHANNELS_PER_WARP],
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>
                acc_frag[BlockConfig_::OUT_CHANNELS_PER_WARP]
                        [BlockConfig_::OH_PER_WARP]) {
    if (!last_slice) {
        gbl2smem_data_visitor.inc_stage();
        gbl2smem_filter_visitor.inc_stage();
        gbl2smem_data_visitor.copy();
        gbl2smem_filter_visitor.copy();
    }

    int data_spatial_idx_base = threadIdx.x / WARP_SIZE * WMMA_N;
    int loop_count = 0;
#pragma unroll
    for (; loop_count <
           BlockConfig_::IC_UNROLL_SIZE * ConvConfig_::FH * ConvConfig_::FW - 1;
         loop_count++) {
        calc<BlockConfig_::OUT_CHANNELS_PER_WARP, BlockConfig_::OH_PER_WARP>(
                data_frag[loop_count % 2], filter_frag[loop_count % 2],
                acc_frag);

        int filter_spatial_idx =
                (loop_count + 1) % (ConvConfig_::FH * ConvConfig_::FW);
        int ic_blk = (loop_count + 1) / (ConvConfig_::FH * ConvConfig_::FW);
        int fh = filter_spatial_idx / ConvConfig_::FW;
        int fw = filter_spatial_idx % ConvConfig_::FW;
        int data_spatial_idx =
                data_spatial_idx_base +
                fh * DataCount<ConvConfig_, BlockConfig_>::LANE_SIZE + fw;
        load_share_mem<ConvConfig_, BlockConfig_>(
                data_frag[(loop_count + 1) % 2],
                filter_frag[(loop_count + 1) % 2], gbl2smem_data_visitor,
                gbl2smem_filter_visitor, data_spatial_idx, filter_spatial_idx,
                ic_blk);
    }

    calc<BlockConfig_::OUT_CHANNELS_PER_WARP, BlockConfig_::OH_PER_WARP>(
            data_frag[(loop_count % 2)], filter_frag[(loop_count % 2)],
            acc_frag);
    if (!last_slice) {
        __syncthreads();
        gbl2smem_data_visitor.commit();
        gbl2smem_filter_visitor.commit();
        __syncthreads();
        load_share_mem<ConvConfig_, BlockConfig_>(
                data_frag[0], filter_frag[0], gbl2smem_data_visitor,
                gbl2smem_filter_visitor, data_spatial_idx_base, 0, 0);
    }
}

#if 0
template <bool last_slice, typename ConvConfig_, typename BlockConfig_>
__device__ void consume_slice_no_reg_cache(
        ConvDataGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>&
                gbl2smem_data_visitor,
        ConvFilterGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>&
                gbl2smem_filter_visitor,
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::col_major>
                data_frag[BlockConfig_::OH_PER_WARP],
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::row_major>
                filter_frag[BlockConfig_::OUT_CHANNELS_PER_WARP],
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>
                acc_frag[BlockConfig_::OUT_CHANNELS_PER_WARP]
                        [BlockConfig_::OH_PER_WARP]) {
    if (!last_slice) {
        gbl2smem_data_visitor.inc_stage();
        gbl2smem_filter_visitor.inc_stage();
        gbl2smem_data_visitor.copy();
        gbl2smem_filter_visitor.copy();
    }

    int data_spatial_idx_base = threadIdx.x / WARP_SIZE * WMMA_N;
    int loop_count = 0;
#pragma unroll
    for (; loop_count <
           BlockConfig_::IC_UNROLL_SIZE * ConvConfig_::FH * ConvConfig_::FW;
         loop_count++) {
        int filter_spatial_idx =
                (loop_count + 0) % (ConvConfig_::FH * ConvConfig_::FW);
        int ic_blk = (loop_count + 0) / (ConvConfig_::FH * ConvConfig_::FW);
        int fh = filter_spatial_idx / ConvConfig_::FW;
        int fw = filter_spatial_idx % ConvConfig_::FW;
        int data_spatial_idx =
                data_spatial_idx_base +
                fh * DataCount<ConvConfig_, BlockConfig_>::LANE_SIZE + fw;

        load_share_mem<ConvConfig_, BlockConfig_>(
                data_frag, filter_frag, gbl2smem_data_visitor,
                gbl2smem_filter_visitor, data_spatial_idx, filter_spatial_idx,
                ic_blk);
        calc<BlockConfig_::OUT_CHANNELS_PER_WARP, BlockConfig_::OH_PER_WARP>(
                data_frag, filter_frag, acc_frag);
    }

    if (!last_slice) {
        __syncthreads();
        gbl2smem_data_visitor.commit();
        gbl2smem_filter_visitor.commit();
        __syncthreads();
    }
}
#endif

template <typename ConvConfig_, typename BlockConfig_>
__global__ void convolution_template_device_u4(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ filter,
        int32_t* __restrict__ out, int N, int IH, int IW, int OH, int OW,
        int PH, int PW, int IC, int OC, int32_t zero) {
    constexpr size_t IC_BLKS = BlockConfig_::IC_BLKS;
    constexpr size_t OUT_CHANNELS_PER_BLOCK =
            FilterCount<ConvConfig_, BlockConfig_>::OUT_CHANNELS_PER_BLOCK;

    const int blocks_per_row = (OW + WMMA_N * BlockConfig_::WARPS_W - 1) /
                               (WMMA_N * BlockConfig_::WARPS_W);
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;
    const int b_oh = bidx / blocks_per_row * BlockConfig_::OH_PER_WARP;
    const int b_ow = bidx % blocks_per_row * (WMMA_N * BlockConfig_::WARPS_W);
    const int warp_x = threadIdx.x / WARP_SIZE;
    const int warp_y = threadIdx.y;

    const int oc_start = bidy * OUT_CHANNELS_PER_BLOCK + warp_y * WMMA_M;
    const int ow_start = b_ow + warp_x * WMMA_N;
    const int b_ih = b_oh * ConvConfig_::SH - PH;
    const int b_iw = b_ow * ConvConfig_::SW - PW;

    const uint8_t* __restrict__ g_ptr_data =
            data + bidz * IC * IH * IW / 2 + (b_ih * IW + b_iw) * 8 / 2;
    const uint8_t* __restrict__ g_ptr_filter =
            filter + bidy * OUT_CHANNELS_PER_BLOCK * ConvConfig_::FH *
                             ConvConfig_::FW * IC / 2;
    const int co_remain = OC - bidy * OUT_CHANNELS_PER_BLOCK;
    int32_t* __restrict__ g_ptr_out = out + bidz * OC * OH * OW +
                                      oc_start * OH * OW +
                                      (b_oh * OW + ow_start) * WMMA_M;
    const int icb = IC / 8;

    __shared__ uint8_t
            smem_data[DataCount<ConvConfig_, BlockConfig_>::SMEM_DATA_ROW]
                     [DataCount<ConvConfig_, BlockConfig_>::SMEM_DATA_COL];
    __shared__ uint8_t smem_filter
            [FilterCount<ConvConfig_, BlockConfig_>::SMEM_FILTER_ROW]
            [FilterCount<ConvConfig_, BlockConfig_>::SMEM_FILTER_COL];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>
            acc_frag[BlockConfig_::OUT_CHANNELS_PER_WARP]
                    [BlockConfig_::OH_PER_WARP];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4, wmma::col_major>
            data_frag[2][BlockConfig_::OH_PER_WARP];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4, wmma::row_major>
            filter_frag[2][BlockConfig_::OUT_CHANNELS_PER_WARP];

    ConvDataGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>
            gbl2smem_data_visitor{smem_data[0], g_ptr_data, IH,  IW,
                                  b_ih,         b_iw,       zero};
    ConvFilterGlobal2ShareMemVisitor<ConvConfig_, BlockConfig_>
            gbl2smem_filter_visitor{smem_filter[0], g_ptr_filter,
                                    IC / 2 * ConvConfig_::FH * ConvConfig_::FW,
                                    co_remain, icb};

#pragma unroll
    for (int i = 0; i < BlockConfig_::OUT_CHANNELS_PER_WARP; ++i) {
#pragma unroll
        for (int j = 0; j < BlockConfig_::OH_PER_WARP; ++j) {
            wmma::fill_fragment(acc_frag[i][j], 0);
        }
    }

    gbl2smem_data_visitor.copy();
    gbl2smem_filter_visitor.copy();
    gbl2smem_data_visitor.commit();
    gbl2smem_filter_visitor.commit();
    __syncthreads();

    load_share_mem<ConvConfig_, BlockConfig_>(
            data_frag[0], filter_frag[0], gbl2smem_data_visitor,
            gbl2smem_filter_visitor, warp_x * WMMA_N, 0, 0);

    int ic_blocks = (icb + IC_BLKS - 1) / IC_BLKS - 1;
#pragma unroll
    for (int ci_blk = 0; ci_blk < ic_blocks; ci_blk++) {
        consume_slice<false, ConvConfig_, BlockConfig_>(
                gbl2smem_data_visitor, gbl2smem_filter_visitor, data_frag,
                filter_frag, acc_frag);
    }
    consume_slice<true, ConvConfig_, BlockConfig_>(
            gbl2smem_data_visitor, gbl2smem_filter_visitor, data_frag,
            filter_frag, acc_frag);

    // store
#pragma unroll
    for (int i = 0; i < BlockConfig_::OUT_CHANNELS_PER_WARP; ++i) {
#pragma unroll
        for (int j = 0; j < BlockConfig_::OH_PER_WARP; ++j) {
            if (b_oh + j < OH &&
                oc_start + i * BlockConfig_::WARPS_OC * WMMA_M < OC &&
                ow_start < OW) {
                wmma::store_matrix_sync(&g_ptr_out[i * BlockConfig_::WARPS_OC *
                                                           WMMA_M * OH * OW +
                                                   j * OW * WMMA_M],
                                        acc_frag[i][j], WMMA_M,
                                        wmma::mem_col_major);
            }
        }
    }
}
#else
template <typename ConvConfig_, typename BlockConfig_>
__global__ void convolution_template_device_u4(
        const uint8_t* __restrict__ /* data */,
        const uint8_t* __restrict__ /* filter */,
        int32_t* __restrict__ /* out */, int /* N */, int /* IH */,
        int /* IW */, int /* OH */, int /* OW */, int /* PH */, int /* PW */,
        int /* IC */, int /* OC */, int32_t /* zero */) {}
#endif
}  // namespace wmma_conv_integer_subbyte_fhxfw

using namespace wmma_conv_integer_subbyte_fhxfw;

void megdnn::cuda::wmma_conv_integer_subbyte::
        _do_wmma_conv_integer_subbyte_fhxfw(
                const uint8_t* d_data, const uint8_t* d_filter, int32_t* d_out,
                int batch_size, int hi, int wi, int ho, int wo, int ph, int pw,
                int ci, int co, int fh, int fw, int sh, int sw, uint8_t zp_data,
                cudaStream_t stream) {
    cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cuda_check(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    zp_data = (zp_data << 4) | zp_data;
    int32_t zero = (zp_data << 24) | (zp_data << 16) | (zp_data << 8) | zp_data;
    if (fh == 3 && fw == 3 && sh == 1 && sw == 1) {
        constexpr size_t warps_w = 2;
        constexpr size_t warps_oc = 4;
        constexpr size_t out_channels_per_warp = 2;
        constexpr size_t oh_per_warp = 8;
        constexpr size_t ic_unroll_size = 2;

        dim3 gridDim;
        dim3 blockDim;
        int blocks_per_row = (wo + WMMA_N * warps_w - 1) / (WMMA_N * warps_w);
        int blocks_per_col = (ho + oh_per_warp - 1) / (oh_per_warp);
        int blocks_per_out_channel =
                (co + WMMA_M * warps_oc * out_channels_per_warp - 1) /
                (WMMA_M * warps_oc * out_channels_per_warp);

        blockDim.x = WARP_SIZE * warps_w;
        blockDim.y = warps_oc;
        blockDim.z = 1;

        gridDim.x = blocks_per_row * blocks_per_col;
        gridDim.y = blocks_per_out_channel;
        gridDim.z = batch_size;

        convolution_template_device_u4<
                ConvConfig<3, 3, 1, 1>,
                BlockConfig<warps_w, warps_oc, out_channels_per_warp,
                            oh_per_warp, ic_unroll_size>>
                <<<gridDim, blockDim, 0, stream>>>(d_data, d_filter, d_out,
                                                   batch_size, hi, wi, ho, wo,
                                                   ph, pw, ci, co, zero);
    } else if (fh == 5 && fw == 5 && sh == 1 && sw == 1) {
        constexpr size_t warps_w = 2;
        constexpr size_t warps_oc = 4;
        constexpr size_t out_channels_per_warp = 2;
        constexpr size_t oh_per_warp = 8;
        constexpr size_t ic_unroll_size = 1;

        dim3 gridDim;
        dim3 blockDim;
        int blocks_per_row = (wo + WMMA_N * warps_w - 1) / (WMMA_N * warps_w);
        int blocks_per_col = (ho + oh_per_warp - 1) / (oh_per_warp);
        int blocks_per_out_channel =
                (co + WMMA_M * warps_oc * out_channels_per_warp - 1) /
                (WMMA_M * warps_oc * out_channels_per_warp);

        blockDim.x = WARP_SIZE * warps_w;
        blockDim.y = warps_oc;
        blockDim.z = 1;

        gridDim.x = blocks_per_row * blocks_per_col;
        gridDim.y = blocks_per_out_channel;
        gridDim.z = batch_size;

        convolution_template_device_u4<
                ConvConfig<5, 5, 1, 1>,
                BlockConfig<warps_w, warps_oc, out_channels_per_warp,
                            oh_per_warp, ic_unroll_size>>
                <<<gridDim, blockDim, 0, stream>>>(d_data, d_filter, d_out,
                                                   batch_size, hi, wi, ho, wo,
                                                   ph, pw, ci, co, zero);
    } else if (fh == 7 && fw == 7 && sh == 1 && sw == 1) {
        constexpr size_t warps_w = 2;
        constexpr size_t warps_oc = 2;
        constexpr size_t out_channels_per_warp = 2;
        constexpr size_t oh_per_warp = 4;
        constexpr size_t ic_unroll_size = 1;

        dim3 gridDim;
        dim3 blockDim;
        int blocks_per_row = (wo + WMMA_N * warps_w - 1) / (WMMA_N * warps_w);
        int blocks_per_col = (ho + oh_per_warp - 1) / (oh_per_warp);
        int blocks_per_out_channel =
                (co + WMMA_M * warps_oc * out_channels_per_warp - 1) /
                (WMMA_M * warps_oc * out_channels_per_warp);

        blockDim.x = WARP_SIZE * warps_w;
        blockDim.y = warps_oc;
        blockDim.z = 1;

        gridDim.x = blocks_per_row * blocks_per_col;
        gridDim.y = blocks_per_out_channel;
        gridDim.z = batch_size;

        convolution_template_device_u4<
                ConvConfig<7, 7, 1, 1>,
                BlockConfig<warps_w, warps_oc, out_channels_per_warp,
                            oh_per_warp, ic_unroll_size>>
                <<<gridDim, blockDim, 0, stream>>>(d_data, d_filter, d_out,
                                                   batch_size, hi, wi, ho, wo,
                                                   ph, pw, ci, co, zero);
    }
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen
