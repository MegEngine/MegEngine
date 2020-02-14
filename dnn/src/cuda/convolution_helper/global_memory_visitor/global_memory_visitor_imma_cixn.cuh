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
 * \file dnn/src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_imma_cixn.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_common.cuh"
#include "src/cuda/convolution_helper/layout.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {

#define MEGDNN_COMMA ,
template <bool check_bounds, typename TileCount_, typename Layout>
struct Global2ShareMemVisitorIMMA_CIxN;

DEF_GLOBAL_MEMORY_VISITOR(Global2ShareMemVisitorIMMA_CIxN, 
        Layout<Format::CHWN4>)
    using IMMAConfig = typename TileCount::IMMAConfig;
    using WarpTileConfig = typename TileCount::WarpTileConfig;
    using ThreadConfig = typename TileCount::ThreadConfig;
    int stride;
    int remain;
    
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * ThreadConfig::nr_thread_x + tidx;
    const int gl_load_y = tid / TileCount::load_x;
    const int gl_load_x = tid - gl_load_y * TileCount::load_x;
    
    copy_t reg[TileCount::reg_h][TileCount::reg_w][TileCount::reg_d];

    __device__ __forceinline__ void init_stride(Layout<Format::CHWN4> layout) {
        stride = layout.channel_stride / TileCount::ldg_load_width;
    }

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                int batch = w_idx * load_width;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
#pragma unroll
                for (int k = 0; k < TileCount::reg_d; ++k) {
                    int channel = ((h_idx * TileCount::reg_d + k));
                    if (check_bounds) {
                        copy_t val = make_zero<copy_t>();
                        if (batch < remain) {
                            val = g_ptr[channel * stride + w_idx];
                        }
                        *(sh_ptr(h_idx, batch * TileCount::reg_d + k)) = val;
                    } else {
                        *(sh_ptr(h_idx, batch * TileCount::reg_d + k)) =
                                g_ptr[channel * stride + w_idx];
                    }
                }
            }
        }
    }

    __device__ __forceinline__ void copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                int batch = w_idx * load_width;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
#pragma unroll
                for (int k = 0; k < TileCount::reg_d; ++k) {
                    int channel = (h_idx * TileCount::reg_d + k);
                    if (check_bounds) {
                        copy_t val = make_zero<copy_t>();
                        if (batch < remain) {
                            val = g_ptr[channel * stride + w_idx];
                        }
                        reg[i][j][k] = val;
                    } else {
                        reg[i][j][k] = g_ptr[channel * stride + w_idx];
                    }
                }
            }
        }
    }
    
    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
#pragma unroll
                for (int k = 0; k < TileCount::reg_d; ++k) {
                    *(sh_ptr(h_idx, w_idx * load_width * TileCount::reg_d +
                                            k)) = reg[i][j][k];
                }
            }
        }
    }
    
    __device__ __forceinline__ int32_t* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_stride + x];
    }
    
    __device__ __forceinline__ void move_forward() {
        g_ptr += WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k / 4 * stride;
    }
};

DEF_GLOBAL_MEMORY_VISITOR(Global2ShareMemVisitorIMMA_CIxN, 
        Layout<Format::CHWN16>) 
    using IMMAConfig = typename TileCount::IMMAConfig;
    using WarpTileConfig = typename TileCount::WarpTileConfig;
    using ThreadConfig = typename TileCount::ThreadConfig;
    int stride;
    int remain;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * ThreadConfig::nr_thread_x + tidx;
    const int gl_load_y = tid / TileCount::load_x;
    const int gl_load_x = tid - gl_load_y * TileCount::load_x;

    copy_t reg[TileCount::reg_h][TileCount::reg_w];
    MEGDNN_STATIC_ASSERT(std::is_same<copy_t MEGDNN_COMMA int4>::value == true,
                         "ldg data type must be int4 for this memory visitor");


    __device__ __forceinline__ void init_stride(Layout<Format::CHWN16> layout) {
        stride = layout.channel_stride / TileCount::ldg_load_width;
    }

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                if (check_bounds) {
                    copy_t val = make_zero<copy_t>();
                    if (w_idx < remain) {
                        val = g_ptr[h_idx * stride + w_idx];
                    }
                    *(sh_ptr_as_copy_t(h_idx, w_idx * load_width)) = val;
                } else {
                    *(sh_ptr_as_copy_t(h_idx, w_idx * load_width)) =
                            g_ptr[h_idx * stride + w_idx];
                }
            }
        }
    }

    __device__ __forceinline__ void copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                if (check_bounds) {
                    copy_t val = make_zero<copy_t>();
                    if (w_idx < remain) {
                        val = g_ptr[h_idx * stride + w_idx];
                    }
                    reg[i][j] = val;
                } else {
                    reg[i][j] = g_ptr[h_idx * stride + w_idx];
                }
            }
        }
    }

    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                *(sh_ptr_as_copy_t(h_idx, w_idx * load_width)) = reg[i][j];
            }
        }
    }

    __device__ __forceinline__ int32_t* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_stride + x];
    }

    __device__ __forceinline__ copy_t* sh_ptr_as_copy_t(int y, int x) {
        return reinterpret_cast<copy_t*>(sh_ptr(y, x));
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += WarpTileConfig::warp_tile_k * stride;
    }
};
#undef MEGDNN_COMMA

}  // namespace cuda
}  // namespace megdnn
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
