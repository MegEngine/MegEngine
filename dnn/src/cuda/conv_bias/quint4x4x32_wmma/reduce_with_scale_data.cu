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
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma/reduce_with_scale_data.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./reduce_with_scale_data.cuh"
#include "./wmma_conv_integer_u4.cuh"
#include "src/cuda/cub/util_ptx.cuh"

using namespace megdnn;
using namespace cuda;
using namespace wmma_conv_integer_subbyte;

namespace {

template <typename ConvConfig, size_t thread_blk_x, size_t thread_blk_y,
          size_t pixels_per_thread_x, size_t pixels_per_thread_y>
struct TileCounter {
    MEGDNN_STATIC_ASSERT(thread_blk_x % WARP_SIZE == 0,
                         "thread block size in dim x not divided by warpSize");
    static const size_t spatial_tile_x = thread_blk_x * pixels_per_thread_x;
    static const size_t spatial_tile_y = thread_blk_y * pixels_per_thread_y;
    static const size_t global_load_tile_x =
            (spatial_tile_x - 1) * ConvConfig::SW + ConvConfig::FW;
    static const size_t global_load_tile_y =
            (spatial_tile_y - 1) * ConvConfig::SH + ConvConfig::FH;
    static const size_t reg_cache_x =
            (global_load_tile_x + WARP_SIZE - 1) / WARP_SIZE;
    static const size_t warps_per_block =
            (thread_blk_x * thread_blk_y) / WARP_SIZE;
    static const size_t reg_cache_y =
            (global_load_tile_y + warps_per_block - 1) / warps_per_block;
    static const size_t smem_stride =
            global_load_tile_x + (global_load_tile_x % 2 == 0);
};

template <typename ConvConfig_, size_t thread_blk_x, size_t thread_blk_y,
          size_t pixels_per_thread_x, size_t pixels_per_thread_y>
__global__ void reduce_in_spatial_block_and_along_input_channel_with_scale_u4(
        int32_t* __restrict__ dst, const uint8_t* __restrict__ src, int IC,
        int IH, int IW, int OH, int OW, int PH, int PW, int32_t scale,
        int32_t zero) {
    typedef TileCounter<ConvConfig_, thread_blk_x, thread_blk_y,
                        pixels_per_thread_x, pixels_per_thread_y>
            TileCounter_;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int oh_start = bidy * TileCounter_::spatial_tile_y;
    const int ow_start = bidx * TileCounter_::spatial_tile_x;
    const int ih_base = oh_start * ConvConfig_::SH - PH;
    const int iw_base = ow_start * ConvConfig_::SW - PW;
    const uint8_t* __restrict__ sptr =
            src + bidz * IC * IH * IW / 2 + (ih_base * IW + iw_base) * 4;

    __shared__ uint8_t smem[TileCounter_::global_load_tile_y]
                           [TileCounter_::smem_stride * 4];
    uint32_t reg_cache[TileCounter_::reg_cache_y][TileCounter_::reg_cache_x];
    int32_t acc[pixels_per_thread_y][pixels_per_thread_x];
    int32_t* __restrict__ dptr =
            dst + bidz * OH * OW + ow_start + oh_start * OW;

    const int tid = tidy * thread_blk_x + tidx;
    const int idx_in_warp = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

#pragma unroll
    for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
        for (int j = 0; j < pixels_per_thread_x; ++j) {
            acc[i][j] = 0;
        }
    }

#pragma unroll
    for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
        for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
            int iw = idx_in_warp + j * WARP_SIZE;
            int ih = warp_id + i * TileCounter_::warps_per_block;
            if (ih_base + ih >= 0 && ih_base + ih < IH && iw_base + iw >= 0 &&
                iw_base + iw < IW) {
                reg_cache[i][j] = *(const uint32_t*)(&sptr[(ih * IW + iw) * 4]);
            } else {
                reg_cache[i][j] = zero;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
        for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
            int x = idx_in_warp + j * WARP_SIZE;
            int y = warp_id + i * TileCounter_::warps_per_block;
            if (y < TileCounter_::global_load_tile_y &&
                x < TileCounter_::global_load_tile_x) {
                *(uint32_t*)(&smem[y][x * 4]) = reg_cache[i][j];
            }
        }
    }

    __syncthreads();

    const int ic_blks = (IC + 7) / 8;
#pragma unroll
    for (int c = 0; c < ic_blks; ++c) {
        sptr += IH * IW * 4;
        if (c < ic_blks - 1) {
#pragma unroll
            for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
                for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
                    int iw = idx_in_warp + j * WARP_SIZE;
                    int ih = warp_id + i * TileCounter_::warps_per_block;
                    if (ih_base + ih >= 0 && ih_base + ih < IH &&
                        iw_base + iw >= 0 && iw_base + iw < IW) {
                        reg_cache[i][j] =
                                *(const uint32_t*)(&sptr[(ih * IW + iw) * 4]);
                    } else {
                        reg_cache[i][j] = zero;
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
            for (int j = 0; j < pixels_per_thread_x; ++j) {
                int x = (j * thread_blk_x + tidx) * ConvConfig_::SW;
                int y = (i * thread_blk_y + tidy) * ConvConfig_::SH;
#pragma unroll
                for (int fh = 0; fh < ConvConfig_::FH; ++fh) {
#pragma unroll
                    for (int fw = 0; fw < ConvConfig_::FW; ++fw) {
                        uint32_t sdata =
                                *(uint32_t*)(&smem[y + fh][(x + fw) * 4]);
#pragma unroll
                        for (int r = 0; r < 8; r++) {
                            uint8_t val = (sdata & 0xF);
                            acc[i][j] += val;
                            sdata >>= 4;
                        }
                    }
                }
            }
        }

        if (c < ic_blks - 1) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
                for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
                    int x = idx_in_warp + j * WARP_SIZE;
                    int y = warp_id + i * TileCounter_::warps_per_block;
                    if (y < TileCounter_::global_load_tile_y &&
                        x < TileCounter_::global_load_tile_x) {
                        *(uint32_t*)(&smem[y][x * 4]) = reg_cache[i][j];
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
        for (int j = 0; j < pixels_per_thread_x; ++j) {
            int x = j * thread_blk_x + tidx;
            int y = i * thread_blk_y + tidy;
            if (oh_start + y < OH && ow_start + x < OW) {
                dptr[y * OW + x] = acc[i][j] * scale;
            }
        }
    }
}

template <typename ConvConfig, size_t thread_blk_x, size_t thread_blk_y,
          size_t pixels_per_thread_x, size_t pixels_per_thread_y>
struct LargeChannelTileCounter {
    static const size_t spatial_tile_x = thread_blk_x * pixels_per_thread_x;
    static const size_t spatial_tile_y = pixels_per_thread_y;
    static const size_t global_load_tile_x =
            (spatial_tile_x - 1) * ConvConfig::SW + ConvConfig::FW;
    static const size_t global_load_tile_y =
            (spatial_tile_y - 1) * ConvConfig::SH + ConvConfig::FH;
    static const size_t reg_cache_x =
            (global_load_tile_x + WARP_SIZE - 1) / WARP_SIZE;
    static const size_t warps_per_block =
            (thread_blk_x * thread_blk_y) / WARP_SIZE;
    static const size_t reg_cache_y =
            (global_load_tile_y * thread_blk_y + warps_per_block - 1) /
            warps_per_block;
    static const size_t smem_stride =
            global_load_tile_x + (global_load_tile_x % 2 == 0);
    static const size_t reduce_dim_0 = thread_blk_y;
    static const size_t reduce_dim_1 = pixels_per_thread_y;
    static const size_t reduce_dim_2 = thread_blk_x * pixels_per_thread_x;
};

template <typename ConvConfig_, size_t thread_blk_x, size_t thread_blk_y,
          size_t pixels_per_thread_x, size_t pixels_per_thread_y>
__global__ void
reduce_in_spatial_block_and_along_input_channel_with_scale_u4_large_channels(
        int32_t* __restrict__ dst, const uint8_t* __restrict__ src, int IC,
        int IH, int IW, int OH, int OW, int PH, int PW, int32_t scale,
        int32_t zero) {
    typedef LargeChannelTileCounter<ConvConfig_, thread_blk_x, thread_blk_y,
                                    pixels_per_thread_x, pixels_per_thread_y>
            TileCounter_;

    const int bidx = blockIdx.x;
    const int bidz = blockIdx.z;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int blocks_per_row = (OW + TileCounter_::spatial_tile_x - 1) /
                               TileCounter_::spatial_tile_x;
    const int bidw = bidx % blocks_per_row;
    const int bidh = bidx / blocks_per_row;

    const int oh_start = bidh * TileCounter_::spatial_tile_y;
    const int ow_start = bidw * TileCounter_::spatial_tile_x;
    const int ih_base = oh_start * ConvConfig_::SH - PH;
    const int iw_base = ow_start * ConvConfig_::SW - PW;
    const uint8_t* __restrict__ sptr =
            src + bidz * IC * IH * IW / 2 + (ih_base * IW + iw_base) * 4;

    __shared__ uint8_t smem[thread_blk_y][TileCounter_::global_load_tile_y]
                           [TileCounter_::smem_stride * 4];
    __shared__ int32_t
            s_reduce[TileCounter_::reduce_dim_0][TileCounter_::reduce_dim_1]
                    [TileCounter_::reduce_dim_2 + 1];
    uint32_t reg_cache[TileCounter_::reg_cache_y][TileCounter_::reg_cache_x];
    int32_t acc[pixels_per_thread_y][pixels_per_thread_x];

    int32_t* __restrict__ dptr =
            dst + bidz * OH * OW + ow_start + oh_start * OW;

    const int tid = tidy * thread_blk_x + tidx;
    const int idx_in_warp = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int ic_blks = IC / 8;

#pragma unroll
    for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
        for (int j = 0; j < pixels_per_thread_x; ++j) {
            acc[i][j] = 0;
        }
    }

#pragma unroll
    for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
        for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
            int iw = idx_in_warp + j * WARP_SIZE;
            int hc = warp_id + i * TileCounter_::warps_per_block;
            int ih = hc % TileCounter_::global_load_tile_y;
            int ic_blk = hc / TileCounter_::global_load_tile_y;
            if (ih_base + ih >= 0 && ih_base + ih < IH && iw_base + iw >= 0 &&
                iw_base + iw < IW) {
                reg_cache[i][j] = 0;
                if (ic_blk < ic_blks)
                    reg_cache[i][j] =
                            *(const uint32_t*)(&sptr[(ic_blk * IH * IW +
                                                      ih * IW + iw) *
                                                     4]);
            } else {
                reg_cache[i][j] = (ic_blk < ic_blks) ? zero : 0;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
        for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
            int x = idx_in_warp + j * WARP_SIZE;
            int hc = warp_id + i * TileCounter_::warps_per_block;
            int ih = hc % TileCounter_::global_load_tile_y;
            int ic_blk = hc / TileCounter_::global_load_tile_y;
            if (ic_blk < thread_blk_y && x < TileCounter_::global_load_tile_x) {
                *(uint32_t*)(&smem[ic_blk][ih][x * 4]) = reg_cache[i][j];
            }
        }
    }

    __syncthreads();

    int blks = (ic_blks + thread_blk_y - 1) / thread_blk_y;
#pragma unroll
    for (int c = 0; c < blks; ++c) {
        sptr += IH * IW * thread_blk_y * 4;
        if (c < blks - 1) {
#pragma unroll
            for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
                for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
                    int iw = idx_in_warp + j * WARP_SIZE;
                    int hc = warp_id + i * TileCounter_::warps_per_block;
                    int ih = hc % TileCounter_::global_load_tile_y;
                    int ic_blk = hc / TileCounter_::global_load_tile_y;
                    int g_ic_blk = ic_blk + c * thread_blk_y;
                    if (ih_base + ih >= 0 && ih_base + ih < IH &&
                        iw_base + iw >= 0 && iw_base + iw < IW) {
                        reg_cache[i][j] = 0;
                        if (g_ic_blk < ic_blks)
                            reg_cache[i][j] =
                                    *(const uint32_t*)(&sptr[(ic_blk * IH * IW +
                                                              ih * IW + iw) *
                                                             4]);
                    } else {
                        reg_cache[i][j] = (g_ic_blk < ic_blks) ? zero : 0;
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
            for (int j = 0; j < pixels_per_thread_x; ++j) {
                int x = (j * thread_blk_x + tidx) * ConvConfig_::SW;
                int y = i * ConvConfig_::SH;
#pragma unroll
                for (int fh = 0; fh < ConvConfig_::FH; ++fh) {
#pragma unroll
                    for (int fw = 0; fw < ConvConfig_::FW; ++fw) {
                        uint32_t sdata =
                                *(uint32_t*)(&smem[tidy][y + fh][(x + fw) * 4]);
#pragma unroll
                        for (int r = 0; r < 8; r++) {
                            uint8_t val = (sdata & 0xF);
                            acc[i][j] += val;
                            sdata >>= 4;
                        }
                    }
                }
            }
        }

        if (c < blks - 1) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < TileCounter_::reg_cache_y; ++i) {
#pragma unroll
                for (int j = 0; j < TileCounter_::reg_cache_x; ++j) {
                    int x = idx_in_warp + j * WARP_SIZE;
                    int hc = warp_id + i * TileCounter_::warps_per_block;
                    int ih = hc % TileCounter_::global_load_tile_y;
                    int ic_blk = hc / TileCounter_::global_load_tile_y;
                    if (ic_blk < thread_blk_y &&
                        x < TileCounter_::global_load_tile_x) {
                        *(uint32_t*)(&smem[ic_blk][ih][x * 4]) =
                                reg_cache[i][j];
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
        for (int j = 0; j < pixels_per_thread_x; ++j) {
            s_reduce[tidy][i][tidx + j * thread_blk_x] = acc[i][j];
        }
    }

    const int nr_ty_per_warp = WARP_SIZE / thread_blk_x;
#pragma unroll
    for (int k = (thread_blk_y >> 1); k; k >>= 1) {
        if (k >= nr_ty_per_warp) {
            __syncthreads();
        } else {
            cub::WARP_SYNC(0xffffffff);
        }
        if (tidy < k) {
#pragma unroll
            for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
                for (int j = 0; j < pixels_per_thread_x; ++j) {
                    s_reduce[tidy][i][tidx + j * thread_blk_x] +=
                            s_reduce[tidy + k][i][tidx + j * thread_blk_x];
                }
            }
        }
    }

    if (tidy == 0) {
#pragma unroll
        for (int i = 0; i < pixels_per_thread_y; ++i) {
#pragma unroll
            for (int j = 0; j < pixels_per_thread_x; ++j) {
                int x = j * thread_blk_x + tidx;
                int y = i;
                if (oh_start + y < OH && ow_start + x < OW) {
                    dptr[y * OW + x] =
                            s_reduce[0][i][tidx + j * thread_blk_x] * scale;
                }
            }
        }
    }
}

}  // namespace

void megdnn::cuda::_do_dispatch_reduce_with_scale_data_u4(
        int32_t* dst, const uint8_t* src, int batch_size, int ih, int iw,
        int oh, int ow, int ph, int pw, int fh, int fw, int sh, int sw, int ic,
        int32_t scale, uint8_t zp_data, cudaStream_t stream) {
    zp_data = (zp_data << 4) | zp_data;
    int32_t zero = (zp_data << 24) | (zp_data << 16) | (zp_data << 8) | zp_data;
    if (fh == 3 && fw == 3 && sh == 1 && sw == 1) {
        typedef ConvConfig<3, 3, 1, 1> ConvConfig_;
        if (ic <= 32 && iw >= 128) {
            constexpr size_t thread_blk_x_ = WARP_SIZE;
            constexpr size_t thread_blk_y_ = 2;
            constexpr size_t pixels_per_thread_x_ = 4;
            constexpr size_t pixels_per_thread_y_ = 2;

            typedef TileCounter<ConvConfig_, thread_blk_x_, thread_blk_y_,
                                pixels_per_thread_x_, pixels_per_thread_y_>
                    TileCounter_;

            dim3 gridDim;
            dim3 blockDim;
            int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                 TileCounter_::spatial_tile_x;
            int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                 TileCounter_::spatial_tile_y;
            blockDim.x = thread_blk_x_;
            blockDim.y = thread_blk_y_;
            gridDim.x = blocks_per_row;
            gridDim.y = blocks_per_col;
            gridDim.z = batch_size;

            reduce_in_spatial_block_and_along_input_channel_with_scale_u4<
                    ConvConfig_, thread_blk_x_, thread_blk_y_,
                    pixels_per_thread_x_, pixels_per_thread_y_>
                    <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw, oh,
                                                       ow, ph, pw, scale, zero);
        } else {
            if (iw <= 32) {
                constexpr size_t thread_blk_x_ = WARP_SIZE / 2;
                constexpr size_t thread_blk_y_ = 8;
                constexpr size_t pixels_per_thread_x_ = 1;
                constexpr size_t pixels_per_thread_y_ = 4;

                typedef LargeChannelTileCounter<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        TileCounter_;

                dim3 gridDim;
                dim3 blockDim;
                int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                     TileCounter_::spatial_tile_x;
                int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                     TileCounter_::spatial_tile_y;
                blockDim.x = thread_blk_x_;
                blockDim.y = thread_blk_y_;
                gridDim.x = blocks_per_row * blocks_per_col;
                gridDim.y = 1;
                gridDim.z = batch_size;

                reduce_in_spatial_block_and_along_input_channel_with_scale_u4_large_channels<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw,
                                                           oh, ow, ph, pw,
                                                           scale, zero);
            } else {
                constexpr size_t thread_blk_x_ = WARP_SIZE / 2;
                constexpr size_t thread_blk_y_ = 4;
                constexpr size_t pixels_per_thread_x_ = 4;
                constexpr size_t pixels_per_thread_y_ = 4;

                typedef LargeChannelTileCounter<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        TileCounter_;

                dim3 gridDim;
                dim3 blockDim;
                int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                     TileCounter_::spatial_tile_x;
                int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                     TileCounter_::spatial_tile_y;
                blockDim.x = thread_blk_x_;
                blockDim.y = thread_blk_y_;
                gridDim.x = blocks_per_row * blocks_per_col;
                gridDim.y = 1;
                gridDim.z = batch_size;

                reduce_in_spatial_block_and_along_input_channel_with_scale_u4_large_channels<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw,
                                                           oh, ow, ph, pw,
                                                           scale, zero);
            }
        }
    } else if (fh == 5 && fw == 5 && sh == 1 && sw == 1) {
        typedef ConvConfig<5, 5, 1, 1> ConvConfig_;
        if (ic <= 32 && iw >= 128) {
            constexpr size_t thread_blk_x_ = WARP_SIZE;
            constexpr size_t thread_blk_y_ = 2;
            constexpr size_t pixels_per_thread_x_ = 4;
            constexpr size_t pixels_per_thread_y_ = 2;

            typedef TileCounter<ConvConfig_, thread_blk_x_, thread_blk_y_,
                                pixels_per_thread_x_, pixels_per_thread_y_>
                    TileCounter_;

            dim3 gridDim;
            dim3 blockDim;
            int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                 TileCounter_::spatial_tile_x;
            int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                 TileCounter_::spatial_tile_y;
            blockDim.x = thread_blk_x_;
            blockDim.y = thread_blk_y_;
            gridDim.x = blocks_per_row;
            gridDim.y = blocks_per_col;
            gridDim.z = batch_size;

            reduce_in_spatial_block_and_along_input_channel_with_scale_u4<
                    ConvConfig_, thread_blk_x_, thread_blk_y_,
                    pixels_per_thread_x_, pixels_per_thread_y_>
                    <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw, oh,
                                                       ow, ph, pw, scale, zero);
        } else {
            if (iw <= 32) {
                constexpr size_t thread_blk_x_ = WARP_SIZE / 2;
                constexpr size_t thread_blk_y_ = 8;
                constexpr size_t pixels_per_thread_x_ = 1;
                constexpr size_t pixels_per_thread_y_ = 4;

                typedef LargeChannelTileCounter<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        TileCounter_;

                dim3 gridDim;
                dim3 blockDim;
                int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                     TileCounter_::spatial_tile_x;
                int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                     TileCounter_::spatial_tile_y;
                blockDim.x = thread_blk_x_;
                blockDim.y = thread_blk_y_;
                gridDim.x = blocks_per_row * blocks_per_col;
                gridDim.y = 1;
                gridDim.z = batch_size;

                reduce_in_spatial_block_and_along_input_channel_with_scale_u4_large_channels<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw,
                                                           oh, ow, ph, pw,
                                                           scale, zero);

            } else {
                constexpr size_t thread_blk_x_ = WARP_SIZE / 2;
                constexpr size_t thread_blk_y_ = 4;
                constexpr size_t pixels_per_thread_x_ = 4;
                constexpr size_t pixels_per_thread_y_ = 4;

                typedef LargeChannelTileCounter<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        TileCounter_;

                dim3 gridDim;
                dim3 blockDim;
                int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                     TileCounter_::spatial_tile_x;
                int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                     TileCounter_::spatial_tile_y;
                blockDim.x = thread_blk_x_;
                blockDim.y = thread_blk_y_;
                gridDim.x = blocks_per_row * blocks_per_col;
                gridDim.y = 1;
                gridDim.z = batch_size;

                reduce_in_spatial_block_and_along_input_channel_with_scale_u4_large_channels<
                        ConvConfig_, thread_blk_x_, thread_blk_y_,
                        pixels_per_thread_x_, pixels_per_thread_y_>
                        <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw,
                                                           oh, ow, ph, pw,
                                                           scale, zero);
            }
        }
    } else if (fh == 7 && fw == 7 && sh == 1 && sw == 1) {
        typedef ConvConfig<7, 7, 1, 1> ConvConfig_;
        if (ic <= 32 && iw >= 128) {
            constexpr size_t thread_blk_x_ = WARP_SIZE;
            constexpr size_t thread_blk_y_ = 2;
            constexpr size_t pixels_per_thread_x_ = 4;
            constexpr size_t pixels_per_thread_y_ = 2;

            typedef TileCounter<ConvConfig_, thread_blk_x_, thread_blk_y_,
                                pixels_per_thread_x_, pixels_per_thread_y_>
                    TileCounter_;

            dim3 gridDim;
            dim3 blockDim;
            int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                 TileCounter_::spatial_tile_x;
            int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                 TileCounter_::spatial_tile_y;
            blockDim.x = thread_blk_x_;
            blockDim.y = thread_blk_y_;
            gridDim.x = blocks_per_row;
            gridDim.y = blocks_per_col;
            gridDim.z = batch_size;

            reduce_in_spatial_block_and_along_input_channel_with_scale_u4<
                    ConvConfig_, thread_blk_x_, thread_blk_y_,
                    pixels_per_thread_x_, pixels_per_thread_y_>
                    <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw, oh,
                                                       ow, ph, pw, scale, zero);
        } else {
            constexpr size_t thread_blk_x_ = WARP_SIZE / 2;
            constexpr size_t thread_blk_y_ = 8;
            constexpr size_t pixels_per_thread_x_ = 1;
            constexpr size_t pixels_per_thread_y_ = 4;

            typedef LargeChannelTileCounter<ConvConfig_, thread_blk_x_,
                                            thread_blk_y_, pixels_per_thread_x_,
                                            pixels_per_thread_y_>
                    TileCounter_;

            dim3 gridDim;
            dim3 blockDim;
            int blocks_per_row = (ow + TileCounter_::spatial_tile_x - 1) /
                                 TileCounter_::spatial_tile_x;
            int blocks_per_col = (oh + TileCounter_::spatial_tile_y - 1) /
                                 TileCounter_::spatial_tile_y;
            blockDim.x = thread_blk_x_;
            blockDim.y = thread_blk_y_;
            gridDim.x = blocks_per_row * blocks_per_col;
            gridDim.y = 1;
            gridDim.z = batch_size;

            reduce_in_spatial_block_and_along_input_channel_with_scale_u4_large_channels<
                    ConvConfig_, thread_blk_x_, thread_blk_y_,
                    pixels_per_thread_x_, pixels_per_thread_y_>
                    <<<gridDim, blockDim, 0, stream>>>(dst, src, ic, ih, iw, oh,
                                                       ow, ph, pw, scale, zero);
        }
    }
    after_kernel_launch();
}

// vim: ft=cpp syntax=cuda.doxygen
