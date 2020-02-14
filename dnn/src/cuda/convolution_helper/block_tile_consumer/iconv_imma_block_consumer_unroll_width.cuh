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
 * \file dnn/src/cuda/convolution_helper/block_tile_consumer/iconv_imma_block_consumer_unroll_width.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {
template <typename Conv1dConfig_, typename IMMAConfig_,
          typename WarpTileConfig_, typename ThreadConfig_>
struct IConvIMMABlockConsumerUnrollWidth {
    using Conv1dConfig = Conv1dConfig_;
    using IMMAConfig = IMMAConfig_;
    using WarpTileConfig = WarpTileConfig_;
    using ThreadConfig = ThreadConfig_;

#if __CUDA_ARCH__ >= 730
    typename IMMAConfig::fragment_b frag_src[WarpTileConfig::warp_tile_n][2];
    typename IMMAConfig::fragment_a frag_filter[WarpTileConfig::warp_tile_m][2];
    typename IMMAConfig::fragment_c frag_acc[WarpTileConfig::warp_tile_m]
                                            [WarpTileConfig::warp_tile_n];
#endif

    __device__ __forceinline__ void init_accumulator() {
#if __CUDA_ARCH__ >= 730
#pragma unroll
        for (int i = 0; i < WarpTileConfig::warp_tile_m; ++i) {
#pragma unroll
            for (int j = 0; j < WarpTileConfig::warp_tile_n; ++j) {
                wmma::fill_fragment(frag_acc[i][j], 0.f);
            }
        }
#endif
    }

#if __CUDA_ARCH__ >= 730
    template <typename DataGlobal2ShareMemVisitor,
              typename FilterGlobal2ShareMemVisitor>
    __device__ __forceinline__ void consume_block(
            DataGlobal2ShareMemVisitor data_gl2sh_visitor,
            FilterGlobal2ShareMemVisitor filter_gl2sh_visitor) {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const int warpx = tidx / ThreadConfig::warp_size;
        const int warpy = tidy;

        static bool const consecutive_width_tile =
                !(WarpTileConfig::warp_tile_n & 0x1);
        if (consecutive_width_tile) {
#pragma unroll
            for (int i = 0; i < (WarpTileConfig::warp_tile_n >> 1); ++i) {
                int i2 = (i << 1);
                int warpx2 = (warpx << 1);
                int32_t* data_sh_ptr = data_gl2sh_visitor.sh_ptr(
                        (warpx2 + i2 * ThreadConfig::nr_warp_x) *
                                Conv1dConfig::sw,
                        0);
                wmma::load_matrix_sync(frag_src[i2][0],
                                       reinterpret_cast<int8_t*>(data_sh_ptr),
                                       IMMAConfig::wmma_k);
                wmma::load_matrix_sync(
                        frag_src[i2 + 1][0],
                        reinterpret_cast<int8_t*>(
                                data_sh_ptr +
                                Conv1dConfig::sw *
                                        IMMAConfig::tile_b_sizes_int),
                        IMMAConfig::wmma_k);
            }
        } else {
#pragma unroll
            for (int i = 0; i < WarpTileConfig::warp_tile_n; ++i) {
                int32_t* data_sh_ptr = data_gl2sh_visitor.sh_ptr(
                        (warpx + i * ThreadConfig::nr_warp_x) *
                                Conv1dConfig::sw,
                        0);
                wmma::load_matrix_sync(frag_src[i][0],
                                       reinterpret_cast<int8_t*>(data_sh_ptr),
                                       IMMAConfig::wmma_k);
            }
        }
#pragma unroll
        for (int j = 0; j < WarpTileConfig::warp_tile_m; ++j) {
            int32_t* ker_sh_ptr = filter_gl2sh_visitor.sh_ptr(
                    0, (warpy + j * ThreadConfig::nr_warp_y) *
                               IMMAConfig::tile_a_sizes_int);
            wmma::load_matrix_sync(frag_filter[j][0],
                                   reinterpret_cast<int8_t*>(ker_sh_ptr),
                                   IMMAConfig::wmma_k);
        }

#pragma unroll
        for (int kw = 0; kw < Conv1dConfig::fw; ++kw) {
            const int comp_idx = (kw & 0x1);
            const int load_idx = 1 - comp_idx;
            if (kw != Conv1dConfig::fw - 1) {
                if (consecutive_width_tile) {
#pragma unroll
                    for (int i = 0; i < (WarpTileConfig::warp_tile_n >> 1);
                         ++i) {
                        int i2 = (i << 1);
                        int warpx2 = (warpx << 1);
                        int32_t* data_sh_ptr = data_gl2sh_visitor.sh_ptr(
                                (warpx2 + i2 * ThreadConfig::nr_warp_x) *
                                                Conv1dConfig::sw +
                                        kw + 1,
                                0);
                        wmma::load_matrix_sync(
                                frag_src[i2][load_idx],
                                reinterpret_cast<int8_t*>(data_sh_ptr),
                                IMMAConfig::wmma_k);
                        wmma::load_matrix_sync(
                                frag_src[i2 + 1][load_idx],
                                reinterpret_cast<int8_t*>(
                                        data_sh_ptr +
                                        Conv1dConfig::sw *
                                                IMMAConfig::tile_b_sizes_int),
                                IMMAConfig::wmma_k);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < WarpTileConfig::warp_tile_n; ++i) {
                        int32_t* data_sh_ptr = data_gl2sh_visitor.sh_ptr(
                                (warpx + i * ThreadConfig::nr_warp_x) *
                                                Conv1dConfig::sw +
                                        kw + 1,
                                0);
                        wmma::load_matrix_sync(
                                frag_src[i][load_idx],
                                reinterpret_cast<int8_t*>(data_sh_ptr),
                                IMMAConfig::wmma_k);
                    }
                }
#pragma unroll
                for (int j = 0; j < WarpTileConfig::warp_tile_m; ++j) {
                    int32_t* ker_sh_ptr = filter_gl2sh_visitor.sh_ptr(
                            kw + 1, (warpy + j * ThreadConfig::nr_warp_y) *
                                            IMMAConfig::tile_a_sizes_int);
                    wmma::load_matrix_sync(
                            frag_filter[j][load_idx],
                            reinterpret_cast<int8_t*>(ker_sh_ptr),
                            IMMAConfig::wmma_k);
                }
            }  // end if ci_inner
#pragma unroll
            for (int i = 0; i < WarpTileConfig::warp_tile_m; ++i) {
#pragma unroll
                for (int j = 0; j < WarpTileConfig::warp_tile_n; ++j) {
                    wmma::mma_sync(frag_acc[i][j], frag_filter[i][comp_idx],
                                   frag_src[j][comp_idx], frag_acc[i][j]);
                }
            }
        }  // end for kw
    }
#else
    template <typename DataGlobal2ShareMemVisitor,
              typename FilterGlobal2ShareMemVisitor>
    __device__ __forceinline__ void consume_block(
            DataGlobal2ShareMemVisitor /* data_gl2sh_visitor */,
            FilterGlobal2ShareMemVisitor /* filter_gl2sh_visitor */) {}
#endif
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
