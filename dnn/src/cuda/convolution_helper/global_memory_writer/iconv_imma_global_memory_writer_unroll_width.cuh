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
 * \file dnn/src/cuda/convolution_helper/global_memory_writer/iconv_imma_global_memory_writer_unroll_width.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda.h>
#if CUDA_VERSION >= 10000
#include <mma.h>
#endif

namespace megdnn {
namespace cuda {
namespace convolution {
#if __CUDA_ARCH__ >= 730
using namespace nvcuda;
#endif

template <typename GlobalMemoryStoreCount>
struct IConvIMMAGlobalMemoryWriterUnrollWidth {
    using IMMAConfig = typename GlobalMemoryStoreCount::IMMAConfig;
    using WarpTileConfig = typename GlobalMemoryStoreCount::WarpTileConfig;
    using ThreadConfig = typename GlobalMemoryStoreCount::ThreadConfig;
    using st_type = typename GlobalMemoryStoreCount::copy_t;
    static constexpr bool consecutive_width_tile =
            GlobalMemoryStoreCount::consecutive_width_tile;
    static constexpr int pack_size = WarpTileConfig::pack_size;

    int32_t* smem;
    float alpha;
    float beta;
    int block_batch_remain;
    int block_out_channel_remain;

    __device__ __forceinline__ void init(int32_t* smem_, const float alpha_,
                                         const float beta_) {
        smem = smem_;
        alpha = alpha_, beta = beta_;
    }

    template <bool check_bounds, typename BiasVisitor, typename Epilogue,
              typename BlockConsumer>
    __device__ __forceinline__ void write(BiasVisitor bias, Epilogue epilogue,
                                          BlockConsumer block_consumer) {
#if __CUDA_ARCH__ >= 730
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int warpx = tidx / ThreadConfig::warp_size;
        const int warpy = tidy;
        const int idx_intra_warp = tidx & (ThreadConfig::warp_size - 1);

        // store fragment to share memory
        if (consecutive_width_tile) {
            const int warpx2 = (warpx << 1);
            int32_t* st_sh_frag_ptr =
                    smem +
                    (warpy * ThreadConfig::nr_warp_x + warpx) *
                            (IMMAConfig::wmma_m * IMMAConfig::wmma_n << 1);
#pragma unroll
            for (int i = 0; i < WarpTileConfig::warp_tile_m; ++i) {
#pragma urnoll
                for (int j = 0; j < (WarpTileConfig::warp_tile_n >> 1); ++j) {
                    int j2 = (j << 1);
                    static int const wmma_n2 = (IMMAConfig::wmma_n << 1);
                    wmma::store_matrix_sync(st_sh_frag_ptr,
                                            block_consumer.frag_acc[i][j2],
                                            wmma_n2, wmma::mem_row_major);
                    wmma::store_matrix_sync(st_sh_frag_ptr + IMMAConfig::wmma_n,
                                            block_consumer.frag_acc[i][j2 + 1],
                                            wmma_n2, wmma::mem_row_major);

                    const int sh_st_y =
                            idx_intra_warp / GlobalMemoryStoreCount::store_x;
                    const int sh_st_x =
                            idx_intra_warp -
                            sh_st_y * GlobalMemoryStoreCount::store_x;
                    const int wmma_tile_h_base = (sh_st_y << 2);
                    const int wmma_tile_w =
                            sh_st_x * GlobalMemoryStoreCount::store_width;
                    if (wmma_tile_h_base + 4 <= IMMAConfig::wmma_m) {
                        int const b0 = wmma_tile_w & (IMMAConfig::wmma_n - 1);
                        int const width =
                                (warpx2 + j2 * ThreadConfig::nr_warp_x) +
                                (wmma_tile_w >> IMMAConfig::wmma_n_bit);
                        int const ch = (warpy + i * ThreadConfig::nr_warp_y) *
                                               IMMAConfig::wmma_m +
                                       wmma_tile_h_base;
                        int const b1 = b0 + 1, b2 = b0 + 2, b3 = b0 + 3;

                        st_type lane0 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 0) *
                                                        wmma_n2 +
                                                wmma_tile_w]));
                        st_type lane1 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 1) *
                                                        wmma_n2 +
                                                wmma_tile_w]));
                        st_type lane2 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 2) *
                                                        wmma_n2 +
                                                wmma_tile_w]));
                        st_type lane3 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 3) *
                                                        wmma_n2 +
                                                wmma_tile_w]));

                        float4 f_conv0 = ::make_float4(lane0.x, lane1.x,
                                                       lane2.x, lane3.x);
                        float4 f_conv1 = ::make_float4(lane0.y, lane1.y,
                                                       lane2.y, lane3.y);
                        float4 f_conv2 = ::make_float4(lane0.z, lane1.z,
                                                       lane2.z, lane3.z);
                        float4 f_conv3 = ::make_float4(lane0.w, lane1.w,
                                                       lane2.w, lane3.w);

                        // store to global memory
                        if (!check_bounds) {
                            float4 f_bias0 = bias.at(b0, ch, 0, width);
                            float4 f_bias1 = bias.at(b1, ch, 0, width);
                            float4 f_bias2 = bias.at(b2, ch, 0, width);
                            float4 f_bias3 = bias.at(b3, ch, 0, width);

                            epilogue.apply(alpha, f_conv0, f_conv1, f_conv2,
                                           f_conv3, beta, f_bias0, f_bias1,
                                           f_bias2, f_bias3, b0, ch, 0, width);
                        } else if (ch < block_out_channel_remain) {
                            if ((block_batch_remain & 0x3) == 0 &&
                                b0 + 4 <= block_batch_remain) {
                                float4 f_bias0 = bias.at(b0, ch, 0, width);
                                float4 f_bias1 = bias.at(b1, ch, 0, width);
                                float4 f_bias2 = bias.at(b2, ch, 0, width);
                                float4 f_bias3 = bias.at(b3, ch, 0, width);

                                epilogue.apply(alpha, f_conv0, f_conv1, f_conv2,
                                               f_conv3, beta, f_bias0, f_bias1,
                                               f_bias2, f_bias3, b0, ch, 0,
                                               width);
                            } else {
#define store(_idx)                                                       \
    if (b0 + _idx < block_batch_remain) {                                 \
        float4 f_bias = bias.at(b##_idx, ch, 0, width);                   \
        epilogue.apply(alpha, f_conv##_idx, beta, f_bias, b##_idx, ch, 0, \
                       width);                                            \
    }
                                store(0);
                                store(1);
                                store(2);
                                store(3);
                            }
                        }  // end if check bounds
                    }      // end if store bound
                }          // end j
            }              // end i
        } else {
            int32_t* st_sh_frag_ptr =
                    smem + (warpy * ThreadConfig::nr_warp_x + warpx) *
                                   IMMAConfig::wmma_m * IMMAConfig::wmma_n;

#pragma unroll
            for (int i = 0; i < WarpTileConfig::warp_tile_m; ++i) {
#pragma urnoll
                for (int j = 0; j < WarpTileConfig::warp_tile_n; ++j) {
                    wmma::store_matrix_sync(
                            st_sh_frag_ptr, block_consumer.frag_acc[i][j],
                            IMMAConfig::wmma_n, wmma::mem_row_major);
                    const int sh_st_y =
                            idx_intra_warp / GlobalMemoryStoreCount::store_x;
                    const int sh_st_x =
                            idx_intra_warp -
                            sh_st_y * GlobalMemoryStoreCount::store_x;
                    const int wmma_tile_h_base = (sh_st_y << 2);
                    const int wmma_tile_w =
                            sh_st_x * GlobalMemoryStoreCount::store_width;
                    if (wmma_tile_h_base + 4 <= IMMAConfig::wmma_m) {
                        int const b0 = wmma_tile_w;
                        int const width = warpx + j * ThreadConfig::nr_warp_x;
                        int const ch = (warpy + i * ThreadConfig::nr_warp_y) *
                                               IMMAConfig::wmma_m +
                                       wmma_tile_h_base;
                        int const b1 = b0 + 1, b2 = b0 + 2, b3 = b0 + 3;

                        st_type lane0 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 0) *
                                                        IMMAConfig::wmma_n +
                                                wmma_tile_w]));
                        st_type lane1 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 1) *
                                                        IMMAConfig::wmma_n +
                                                wmma_tile_w]));
                        st_type lane2 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 2) *
                                                        IMMAConfig::wmma_n +
                                                wmma_tile_w]));
                        st_type lane3 = *(reinterpret_cast<st_type*>(
                                &st_sh_frag_ptr[(wmma_tile_h_base + 3) *
                                                        IMMAConfig::wmma_n +
                                                wmma_tile_w]));

                        float4 f_conv0 = ::make_float4(lane0.x, lane1.x,
                                                       lane2.x, lane3.x);
                        float4 f_conv1 = ::make_float4(lane0.y, lane1.y,
                                                       lane2.y, lane3.y);
                        float4 f_conv2 = ::make_float4(lane0.z, lane1.z,
                                                       lane2.z, lane3.z);
                        float4 f_conv3 = ::make_float4(lane0.w, lane1.w,
                                                       lane2.w, lane3.w);

                        // store to global memory
                        if (!check_bounds) {
                            float4 f_bias0 = bias.at(b0, ch, 0, width);
                            float4 f_bias1 = bias.at(b1, ch, 0, width);
                            float4 f_bias2 = bias.at(b2, ch, 0, width);
                            float4 f_bias3 = bias.at(b3, ch, 0, width);

                            epilogue.apply(alpha, f_conv0, f_conv1, f_conv2,
                                           f_conv3, beta, f_bias0, f_bias1,
                                           f_bias2, f_bias3, b0, ch, 0, width);
                        } else if (ch < block_out_channel_remain) {
                            if ((block_batch_remain & 0x3) == 0 &&
                                b0 + 4 <= block_batch_remain) {
                                float4 f_bias0 = bias.at(b0, ch, 0, width);
                                float4 f_bias1 = bias.at(b1, ch, 0, width);
                                float4 f_bias2 = bias.at(b2, ch, 0, width);
                                float4 f_bias3 = bias.at(b3, ch, 0, width);

                                epilogue.apply(alpha, f_conv0, f_conv1, f_conv2,
                                               f_conv3, beta, f_bias0, f_bias1,
                                               f_bias2, f_bias3, b0, ch, 0,
                                               width);
                            } else {
                                store(0);
                                store(1);
                                store(2);
                                store(3);
#undef store
                            }
                        }  // end if check bounds
                    }      // end if store bound
                }          // end j
            }              // end i
        }
#endif
    }
};

}  // namespace cuda
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
