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
 * \file dnn/src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_imma_cixwixn.cuh
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
struct Global2ShareMemVisitorIMMA_CIxWIxN;

DEF_GLOBAL_MEMORY_VISITOR(Global2ShareMemVisitorIMMA_CIxWIxN, 
        Layout<Format::CHWN4>) 
    using IMMAConfig = typename TileCount::IMMAConfig;
    using WarpTileConfig = typename TileCount::WarpTileConfig;
    using ThreadConfig = typename TileCount::ThreadConfig;
    int stride;
    int remain;
    int width_stride;
    int width_start;
    int width_end;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * ThreadConfig::nr_thread_x + tidx;
    const int gl_load_y = tid / TileCount::load_x;
    const int gl_load_x = tid - gl_load_y * TileCount::load_x;

    copy_t reg[TileCount::reg_h][TileCount::reg_w][TileCount::reg_d];
    MEGDNN_STATIC_ASSERT(std::is_same<copy_t MEGDNN_COMMA int4>::value == true,
                         "ldg data type must be int4 for this memory visitor");

    __device__ __forceinline__ void init_stride(Layout<Format::CHWN4> layout) {
        stride = layout.channel_stride / TileCount::ldg_load_width;
        width_stride = layout.width_stride / TileCount::ldg_load_width;
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
                int batch = (w_idx << 2);
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                if (check_bounds) {
                    copy_t c0 = make_zero<copy_t>();
                    copy_t c1 = make_zero<copy_t>();
                    copy_t c2 = make_zero<copy_t>();
                    copy_t c3 = make_zero<copy_t>();
                    if (h_idx >= width_start && h_idx < width_end &&
                        batch < remain) {
                        c0 = g_ptr[0 * stride + h_idx * width_stride + w_idx];
                        c1 = g_ptr[1 * stride + h_idx * width_stride + w_idx];
                        c2 = g_ptr[2 * stride + h_idx * width_stride + w_idx];
                        c3 = g_ptr[3 * stride + h_idx * width_stride + w_idx];
                    }
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4))) =
                            make_int4(c0.x, c1.x, c2.x, c3.x);
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 4)) =
                            make_int4(c0.y, c1.y, c2.y, c3.y);
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 8)) =
                            make_int4(c0.z, c1.z, c2.z, c3.z);
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 12)) =
                            make_int4(c0.w, c1.w, c2.w, c3.w);
                } else {
                    copy_t c0 = make_zero<copy_t>();
                    copy_t c1 = make_zero<copy_t>();
                    copy_t c2 = make_zero<copy_t>();
                    copy_t c3 = make_zero<copy_t>();
                    if (h_idx >= width_start && h_idx < width_end) {
                        c0 = g_ptr[0 * stride + h_idx * width_stride + w_idx];
                        c1 = g_ptr[1 * stride + h_idx * width_stride + w_idx];
                        c2 = g_ptr[2 * stride + h_idx * width_stride + w_idx];
                        c3 = g_ptr[3 * stride + h_idx * width_stride + w_idx];
                    }
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4))) =
                            make_int4(c0.x, c1.x, c2.x, c3.x);
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 4)) =
                            make_int4(c0.y, c1.y, c2.y, c3.y);
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 8)) =
                            make_int4(c0.z, c1.z, c2.z, c3.z);
                    *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 12)) =
                            make_int4(c0.w, c1.w, c2.w, c3.w);
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
                int batch = (w_idx << 2);
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                if (check_bounds) {
                    copy_t c0 = make_zero<copy_t>();
                    copy_t c1 = make_zero<copy_t>();
                    copy_t c2 = make_zero<copy_t>();
                    copy_t c3 = make_zero<copy_t>();
                    if (h_idx >= width_start && h_idx < width_end &&
                        batch < remain) {
                        c0 = g_ptr[0 * stride + h_idx * width_stride + w_idx];
                        c1 = g_ptr[1 * stride + h_idx * width_stride + w_idx];
                        c2 = g_ptr[2 * stride + h_idx * width_stride + w_idx];
                        c3 = g_ptr[3 * stride + h_idx * width_stride + w_idx];
                    }
                    reg[i][j][0] = make_int4(c0.x, c1.x, c2.x, c3.x);
                    reg[i][j][1] = make_int4(c0.y, c1.y, c2.y, c3.y);
                    reg[i][j][2] = make_int4(c0.z, c1.z, c2.z, c3.z);
                    reg[i][j][3] = make_int4(c0.w, c1.w, c2.w, c3.w);
                } else {
                    copy_t c0 = make_zero<copy_t>();
                    copy_t c1 = make_zero<copy_t>();
                    copy_t c2 = make_zero<copy_t>();
                    copy_t c3 = make_zero<copy_t>();
                    if (h_idx >= width_start && h_idx < width_end) {
                        c0 = g_ptr[0 * stride + h_idx * width_stride + w_idx];
                        c1 = g_ptr[1 * stride + h_idx * width_stride + w_idx];
                        c2 = g_ptr[2 * stride + h_idx * width_stride + w_idx];
                        c3 = g_ptr[3 * stride + h_idx * width_stride + w_idx];
                    }
                    reg[i][j][0] = make_int4(c0.x, c1.x, c2.x, c3.x);
                    reg[i][j][1] = make_int4(c0.y, c1.y, c2.y, c3.y);
                    reg[i][j][2] = make_int4(c0.z, c1.z, c2.z, c3.z);
                    reg[i][j][3] = make_int4(c0.w, c1.w, c2.w, c3.w);
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
                *(sh_ptr_as_copy_t(h_idx, (w_idx << 4))) = reg[i][j][0];
                *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 4)) = reg[i][j][1];
                *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 8)) = reg[i][j][2];
                *(sh_ptr_as_copy_t(h_idx, (w_idx << 4) + 12)) = reg[i][j][3];
            }
        }
    }

    __device__ __forceinline__ copy_t* sh_ptr_as_copy_t(int y, int x) {
        return reinterpret_cast<copy_t*>(sh_ptr(y, x));
    }
    __device__ __forceinline__ int32_t* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k / 4 * stride;
    }

    __device__ __forceinline__ void set_range(const int start, const int end) {
        width_start = start, width_end = end;
    }
};
#undef MEGDNN_COMMA

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
