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
 * \file dnn/src/cuda/convolution_helper/config.cuh
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

template <int reg_m_, int reg_n_, int reg_k_, int reg_width_ = 1>
struct RegBlockConfig {
    static int constexpr pack_size = 4;
    static int constexpr pack_size_bit = 2;
    static int constexpr reg_m = reg_m_;
    static int constexpr reg_n = reg_n_;
    static int constexpr reg_k = reg_k_;
    MEGDNN_STATIC_ASSERT(reg_m % pack_size == 0,
                         "reg_m must be a multiple of pack_size");
    MEGDNN_STATIC_ASSERT(reg_k % pack_size == 0,
                         "reg_k must be a multiple of pack_size");
    static int constexpr reg_k_packed = reg_k / pack_size;
    static int constexpr reg_m_packed = reg_m / pack_size;
    static int constexpr reg_width = reg_width_;
};

template <int thread_x, int thread_y>
struct ThreadConfig {
    static int constexpr warp_size = 32;
    static int constexpr nr_thread_x = thread_x;
    static int constexpr nr_thread_y = thread_y;
    static int constexpr nr_threads = nr_thread_x * nr_thread_y;
    static int constexpr nr_warp_x =
            !(nr_thread_x & 0x1f) ? (nr_thread_x >> 5) : 0;
    static int constexpr nr_warp_y = !(nr_thread_x & 0x1f) ? nr_thread_y : 0;
};
static int constexpr WARP_SIZE = ThreadConfig<1, 1>::warp_size;

template <int fw_, int sw_>
struct Conv1dConfig {
    static int constexpr fw = fw_;
    static int constexpr sw = sw_;
};

template <int m_, int n_, int k_>
struct IMMAConfig {
    static int constexpr wmma_m = m_;
    static int constexpr wmma_n = n_;
    static int constexpr wmma_k = k_;
    static int constexpr tile_a_sizes_bytes = wmma_m * wmma_k;
    static int constexpr tile_b_sizes_bytes = wmma_n * wmma_k;
    static int constexpr tile_a_sizes_int = tile_a_sizes_bytes / 4;
    static int constexpr tile_b_sizes_int = tile_b_sizes_bytes / 4;
    static int constexpr tile_c_sizes_int = wmma_m * wmma_n;
    static int constexpr wmma_n_bit = wmma_n == 8 ? 3 : (wmma_n == 16 ? 4 : 5);
    static int constexpr wmma_m_bit = wmma_m == 8 ? 3 : (wmma_m == 16 ? 4 : 5);
#if __CUDA_ARCH__ >= 730
    using fragment_a = wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k,
                                      int8_t, wmma::row_major>;
    using fragment_b = wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k,
                                      int8_t, wmma::col_major>;
    using fragment_c =
            wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, int32_t>;
#endif
};

template <int warp_tile_m_, int warp_tile_n_, int warp_tile_k_>
struct WarpTileConfig {
    static int constexpr warp_tile_m = warp_tile_m_;
    static int constexpr warp_tile_n = warp_tile_n_;
    static int constexpr warp_tile_k = warp_tile_k_;
    static int constexpr pack_size = sizeof(int32_t) / sizeof(int8_t);
    static int constexpr pack_size_bit = 2;
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
