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
 * \file dnn/src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_cixhw.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/arch.h"
#include "src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_common.cuh"
#include "src/cuda/convolution_helper/layout.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {
template <typename TileCount_, typename Layout>
struct Global2ShareMemVisitorBase_CIxHW {
    using TileCount = TileCount_;
    using copy_t = typename TileCount::copy_t;
    using smem_storage_dtype = typename TileCount::smem_storage_dtype;

    using RegBlockConfig = typename TileCount::RegBlockConfig;
    using ThreadConfig = typename TileCount::ThreadConfig;

    const copy_t* __restrict__ g_ptr;
    int stride;
    smem_storage_dtype* smem;

    __device__ Global2ShareMemVisitorBase_CIxHW(smem_storage_dtype* smem_)
            : smem{smem_} {}

    __device__ __forceinline__ void init_stride(Layout layout) {
        stride = layout.channel_stride / TileCount::ldg_load_width;
    }

    __device__ __forceinline__ int32_t* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_stride + x];
    }

    __device__ __forceinline__ copy_t* sh_ptr_as_copy_t(int y, int x) {
        return reinterpret_cast<copy_t*>(sh_ptr(y, x));
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += RegBlockConfig::reg_k_packed * stride;
    }
};

template <bool check_bounds, bool precomp_offset, typename TileCount_,
          typename Layout>
struct Global2ShareMemVisitor_CIxHW;

#define DEF(_precomp_offset, _Layout)                                        \
    template <bool check_bounds, typename TileCount_>                        \
    struct Global2ShareMemVisitor_CIxHW<check_bounds, _precomp_offset,       \
                                        TileCount_, _Layout>                 \
            : public Global2ShareMemVisitorBase_CIxHW<TileCount_, _Layout> { \
        using Base = Global2ShareMemVisitorBase_CIxHW<TileCount_, _Layout>;  \
        using TileCount = typename Base::TileCount;                          \
        using copy_t = typename Base::copy_t;                                \
        using smem_storage_dtype = typename Base::smem_storage_dtype;        \
        using RegBlockConfig = typename TileCount::RegBlockConfig;           \
        using ThreadConfig = typename TileCount::ThreadConfig;               \
        using Base::g_ptr;                                                   \
        using Base::stride;                                                  \
        using Base::smem;                                                    \
        using Base::sh_ptr_as_copy_t;                                        \
        static constexpr int load_width = TileCount::load_width;             \
        static constexpr bool precomp_offset = _precomp_offset;              \
                                                                             \
        const int tidx = threadIdx.x;                                        \
        const int tidy = threadIdx.y;                                        \
        const int tid = tidy * ThreadConfig::nr_thread_x + tidx;             \
        const int gl_load_y = tid / TileCount::load_x;                       \
        const int gl_load_x = tid - gl_load_y * TileCount::load_x;           \
                                                                             \
        const int* __restrict__ offset;                                      \
        int remain;

DEF(true, Layout<NCHW4>) 

    copy_t reg[TileCount::reg_h][TileCount::reg_w][TileCount::reg_d];
    MEGDNN_STATIC_ASSERT(load_width == 4,
                         "load four element from src tensor per time");

    __device__ Global2ShareMemVisitor_CIxHW(smem_storage_dtype* smem_,
                                            const int* __restrict__ offset_)
            : Base{smem_}, offset{offset_} {}

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                int out_offset = w_idx * load_width;
                int4 in_offset =
                        *reinterpret_cast<const int4*>(&offset[out_offset]);
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                copy_t ix = make_zero<copy_t>();
                copy_t iy = ix;
                copy_t iz = ix;
                copy_t iw = ix;
                if (in_offset.x >= 0) {
                    ix = g_ptr[h_idx * stride + in_offset.x];
                }
                if (in_offset.y >= 0) {
                    iy = g_ptr[h_idx * stride + in_offset.y];
                }
                if (in_offset.z >= 0) {
                    iz = g_ptr[h_idx * stride + in_offset.z];
                }
                if (in_offset.w >= 0) {
                    iw = g_ptr[h_idx * stride + in_offset.w];
                }
                *(sh_ptr_as_copy_t(h_idx, out_offset + 0)) = ix;
                *(sh_ptr_as_copy_t(h_idx, out_offset + 1)) = iy;
                *(sh_ptr_as_copy_t(h_idx, out_offset + 2)) = iz;
                *(sh_ptr_as_copy_t(h_idx, out_offset + 3)) = iw;
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
                int out_offset = w_idx * load_width;
                int4 in_offset =
                        *reinterpret_cast<const int4*>(&offset[out_offset]);
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                copy_t ix = make_zero<copy_t>();
                copy_t iy = ix;
                copy_t iz = ix;
                copy_t iw = ix;
                if (in_offset.x >= 0) {
                    ix = g_ptr[h_idx * stride + in_offset.x];
                }
                if (in_offset.y >= 0) {
                    iy = g_ptr[h_idx * stride + in_offset.y];
                }
                if (in_offset.z >= 0) {
                    iz = g_ptr[h_idx * stride + in_offset.z];
                }
                if (in_offset.w >= 0) {
                    iw = g_ptr[h_idx * stride + in_offset.w];
                }
                reg[i][j][0] = ix;
                reg[i][j][1] = iy;
                reg[i][j][2] = iz;
                reg[i][j][3] = iw;
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
                int out_offset = w_idx * load_width;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                *(sh_ptr_as_copy_t(h_idx, out_offset + 0)) = reg[i][j][0];
                *(sh_ptr_as_copy_t(h_idx, out_offset + 1)) = reg[i][j][1];
                *(sh_ptr_as_copy_t(h_idx, out_offset + 2)) = reg[i][j][2];
                *(sh_ptr_as_copy_t(h_idx, out_offset + 3)) = reg[i][j][3];
            }
        }
    }
};
    
DEF(false, Layout<NCHW4>)

    copy_t reg[TileCount::reg_h][TileCount::reg_w];
    __device__ Global2ShareMemVisitor_CIxHW(smem_storage_dtype* smem_)
            : Base{smem_} {}

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_h; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
            if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                continue;
#pragma unroll
            for (int j = 0; j < TileCount::reg_w; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                int spatial = w_idx * load_width;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                if (check_bounds) {
                    copy_t val = make_zero<copy_t>();
                    if (spatial < remain) {
                        val = g_ptr[h_idx * stride + w_idx];
                    }
                    *(sh_ptr_as_copy_t(h_idx, spatial)) = val;
                } else {
                    *(sh_ptr_as_copy_t(h_idx, spatial)) =
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
                int spatial = w_idx * load_width;
                if (TileCount::check_bounds_w &&
                    w_idx >= TileCount::smem_load_x)
                    continue;
                if (check_bounds) {
                    copy_t val = make_zero<copy_t>();
                    if (spatial < remain) {
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
};

#undef DEF

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
