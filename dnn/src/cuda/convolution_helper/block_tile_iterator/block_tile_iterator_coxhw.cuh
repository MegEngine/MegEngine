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
 * \file dnn/src/cuda/convolution_helper/block_tile_iterator/block_tile_iterator_coxhw.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/prologue.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {
template <typename DataTileCount_, typename FilterTileCount_,
          typename Prologue = ConvPrologue>
struct BlockTileIterator_COxHW {
    using DataTileCount = DataTileCount_;
    using FilterTileCount = FilterTileCount_;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    int block_batch;
    int block_out_channel;
    int block_out_height_width;
    int block_out_height;
    int block_out_width;
    int block_out_channel_remain;
    int block_out_height_width_remain;

    template <typename Param>
    __device__ __forceinline__ void init_with_param(const Param& param) {
        block_batch = bidz;
        block_out_height_width =
                bidx * DataTileCount::block_tile_out_height_width;
        block_out_channel = bidy * FilterTileCount::block_tile_out_channel;
        block_out_height = block_out_height_width / param.wo;
        block_out_width = block_out_height_width - block_out_height * param.wo;
        block_out_channel_remain = param.co - block_out_channel;
        block_out_height_width_remain =
                param.ho * param.wo - block_out_height_width;
    }

    template <typename DataGlobal2ShareMemVisitor,
              typename FilterGlobal2ShareMemVisitor>
    __device__ __forceinline__ void set_remain(
            DataGlobal2ShareMemVisitor& src_gl2sh_visitor,
            FilterGlobal2ShareMemVisitor& filter_gl2sh_visitor) {
        if (!DataGlobal2ShareMemVisitor::precomp_offset) {
            src_gl2sh_visitor.remain = block_out_height_width_remain;
        }
        filter_gl2sh_visitor.remain = block_out_channel_remain;
    }

    template <typename GlobalMemoryWriter>
    __device__ __forceinline__ void set_remain(
            GlobalMemoryWriter& global_memory_writer) {
        global_memory_writer.block_out_channel_remain =
                block_out_channel_remain;
        global_memory_writer.block_out_height_width_remain =
                block_out_height_width_remain;
    }

    template <typename InputLayout, typename KernLayout, typename src_dtype,
              typename filter_dtype, typename Param,
              typename DataGlobal2ShareMemVisitor,
              typename FilterGlobal2ShareMemVisitor, typename BlockConsumer>
    __device__ __forceinline__ void iterate_with_param(
            const src_dtype* __restrict__ src,
            const filter_dtype* __restrict__ filter, const Param& param,
            DataGlobal2ShareMemVisitor src_gl2sh_visitor,
            FilterGlobal2ShareMemVisitor filter_gl2sh_visitor,
            BlockConsumer& consumer) {
        Prologue::template prologue(src, filter, param, block_batch,
                                    block_out_channel, block_out_height,
                                    block_out_width);
        static constexpr bool precomp_offset =
                DataGlobal2ShareMemVisitor::precomp_offset;
        InputLayout src_layout;
        KernLayout filter_layout;
        src_layout.init(param.n, param.ci, param.hi, param.wi);
        filter_layout.init(param.co, param.ci, param.fh, param.fw);
        const src_dtype* __restrict__ g_src_ptr;
        if (precomp_offset) {
            g_src_ptr = src + src_layout.offset(block_batch, 0, 0, 0);
        } else {
            g_src_ptr =
                    src + src_layout.offset(block_batch, 0, block_out_height,
                                            block_out_width);
        }
        const filter_dtype* __restrict__ g_filter_ptr =
                filter + filter_layout.offset(block_out_channel, 0, 0, 0);

        src_gl2sh_visitor.init_stride(src_layout);
        filter_gl2sh_visitor.init_stride(filter_layout);

        const int ci_blks =
                (param.ci + DataTileCount::block_tile_in_channel - 1) /
                DataTileCount::block_tile_in_channel;

        if (precomp_offset) {
            src_gl2sh_visitor.offset += block_out_height_width;
        }
        src_gl2sh_visitor.g_ptr = reinterpret_cast<
                const typename DataGlobal2ShareMemVisitor::copy_t*>(g_src_ptr);
        filter_gl2sh_visitor.g_ptr = reinterpret_cast<
                const typename FilterGlobal2ShareMemVisitor::copy_t*>(
                g_filter_ptr);
        src_gl2sh_visitor.first_copy();
        filter_gl2sh_visitor.first_copy();

        __syncthreads();

        const int filter_pixels = param.fh * param.fw;
        const int img_pixels = param.ho * param.wo;

        for (int f = 0; f < filter_pixels; f++) {
            for (int ci_outer = 0; ci_outer < ci_blks; ci_outer++) {
                if (ci_outer == ci_blks - 1) {
                    if (f < filter_pixels - 1) {
                        int f_next = f + 1;
                        int kh = f_next / param.fw;
                        int kw = f_next - kh * param.fw;
                        // rewind
                        if (precomp_offset) {
                            src_gl2sh_visitor.g_ptr = reinterpret_cast<
                                    const typename DataGlobal2ShareMemVisitor::
                                            copy_t*>(g_src_ptr);
                            src_gl2sh_visitor.offset += img_pixels;
                        }
                        filter_gl2sh_visitor.g_ptr = reinterpret_cast<
                                const typename FilterGlobal2ShareMemVisitor::
                                        copy_t*>(
                                g_filter_ptr +
                                filter_layout.offset(0, 0, kh, kw));
                        src_gl2sh_visitor.copy();
                        filter_gl2sh_visitor.copy();
                    }
                } else {
                    src_gl2sh_visitor.move_forward();
                    filter_gl2sh_visitor.move_forward();
                    src_gl2sh_visitor.copy();
                    filter_gl2sh_visitor.copy();
                }

                consumer.consume_block(src_gl2sh_visitor,
                                                filter_gl2sh_visitor);

                if (!(ci_outer == ci_blks - 1 && f == filter_pixels - 1)) {
                    __syncthreads();
                    src_gl2sh_visitor.commit();
                    filter_gl2sh_visitor.commit();
                    __syncthreads();
                }
            }
        }
    }
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
