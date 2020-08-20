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
 * \file dnn/src/cuda/convolution_helper/block_tile_iterator/block_tile_iterator_unroll_width_v2.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

namespace megdnn {
namespace cuda {
namespace convolution {
template <typename DataTileCount_, typename FilterTileCount_>
struct BlockTileIteratorUnrollWidthV2 {
    using DataTileCount = DataTileCount_;
    using FilterTileCount = FilterTileCount_;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    int block_batch;
    int block_out_channel;
    int block_out_height;
    int block_out_width;
    int block_in_width;
    int block_batch_remain;
    int block_out_channel_remain;

    template <typename Param>
    __device__ __forceinline__ void init_with_param(const Param& param) {
        const int blocks_per_image_row =
                (param.wo + DataTileCount::block_tile_out_width - 1) /
                DataTileCount::block_tile_out_width;
        block_out_height = bidx / blocks_per_image_row;
        block_out_width = bidx - blocks_per_image_row * block_out_height;
        block_out_width = block_out_width * DataTileCount::block_tile_out_width;
        block_out_channel = bidz * FilterTileCount::block_tile_out_channel;
        block_batch = bidy * DataTileCount::block_tile_batch;
        block_in_width = block_out_width * param.sw - param.pw;
        block_batch_remain = param.n - block_batch;
        block_out_channel_remain = param.co - block_out_channel;
    }

    template <typename DataGlobal2ShareMemVisitor,
              typename FilterGlobal2ShareMemVisitor>
    __device__ __forceinline__ void set_remain(
            DataGlobal2ShareMemVisitor& src_gl2sh_visitor,
            FilterGlobal2ShareMemVisitor& filter_gl2sh_visitor) {
        src_gl2sh_visitor.remain = block_batch_remain;
        filter_gl2sh_visitor.remain = block_out_channel_remain;
    }

    template <typename GlobalMemoryWriter>
    __device__ __forceinline__ void set_remain(
            GlobalMemoryWriter& global_memory_writer) {
        global_memory_writer.block_batch_remain = block_batch_remain;
        global_memory_writer.block_out_channel_remain =
                block_out_channel_remain;
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
        InputLayout src_layout;
        KernLayout filter_layout;
        src_layout.init(param.n, param.ci, param.hi, param.wi);
        filter_layout.init(param.co, param.ci, param.fh, param.fw);
        const src_dtype* __restrict__ g_src_ptr =
                src + src_layout.offset(block_batch, 0, 0, block_in_width);
        const filter_dtype* __restrict__ g_filter_ptr =
                filter + filter_layout.offset(block_out_channel, 0, 0, 0);
        src_gl2sh_visitor.init_stride(src_layout);
        filter_gl2sh_visitor.init_stride(filter_layout);

        int h_base = block_out_height * param.sh - param.ph;
        int h_start = h_base >= 0 ? h_base : 0;
        int h_end = h_base + param.fh - 1;
        h_end = h_end < param.hi ? h_end : param.hi - 1;

        const int ci_blks =
                (param.ci + DataTileCount::block_tile_in_channel - 1) /
                DataTileCount::block_tile_in_channel;
        int kh = h_start - h_base;

        src_gl2sh_visitor.g_ptr = reinterpret_cast<
                const typename DataGlobal2ShareMemVisitor::copy_t*>(
                g_src_ptr + src_layout.offset(0, 0, h_start, 0));
        filter_gl2sh_visitor.g_ptr = reinterpret_cast<
                const typename FilterGlobal2ShareMemVisitor::copy_t*>(
                g_filter_ptr + filter_layout.offset(0, 0, kh, 0));
        src_gl2sh_visitor.set_range(-block_in_width, param.wi - block_in_width);
        src_gl2sh_visitor.first_copy();
        filter_gl2sh_visitor.first_copy();

        __syncthreads();

        for (int h = h_start; h <= h_end; ++h) {
            for (int ci_outer = 0; ci_outer < ci_blks; ci_outer++) {
                if (ci_outer == ci_blks - 1) {
                    if (h != h_end) {
                        int h_next = h + 1;
                        int kh = h_next - h_base;
                        src_gl2sh_visitor.g_ptr = reinterpret_cast<
                                const typename DataGlobal2ShareMemVisitor::
                                        copy_t*>(
                                g_src_ptr + src_layout.offset(0, 0, h_next, 0));
                        filter_gl2sh_visitor.g_ptr = reinterpret_cast<
                                const typename FilterGlobal2ShareMemVisitor::
                                        copy_t*>(
                                g_filter_ptr +
                                filter_layout.offset(0, 0, kh, 0));
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

                if (!(ci_outer == ci_blks - 1 && h == h_end)) {
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
