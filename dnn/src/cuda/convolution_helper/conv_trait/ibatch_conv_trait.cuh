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
 * \file dnn/src/cuda/convolution_helper/conv_trait/ibatch_conv_trait.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/block_tile_consumer/iconv_block_consumer_coxhw.cuh"
#include "src/cuda/convolution_helper/block_tile_iterator/block_tile_iterator_coxhw.cuh"
#include "src/cuda/convolution_helper/config.cuh"
#include "src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_cixhw.cuh"
#include "src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_cixn.cuh"
#include "src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor_coxci.cuh"
#include "src/cuda/convolution_helper/global_memory_writer/iconv_global_memory_writer_coxhw.cuh"
#include "src/cuda/convolution_helper/layout.cuh"
#include "src/cuda/convolution_helper/parameter.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {
#define COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM(                   \
        _src_dtype, _filter_dtype, _smem_storage_dtype, _input_layout, \
        _kern_layout, _output_layout, _conv_param)                     \
    using src_dtype = _src_dtype;                                      \
    using filter_dtype = _filter_dtype;                                \
    using smem_storage_dtype = _smem_storage_dtype;                    \
    using InputLayout = _input_layout;                                 \
    using KernLayout = _kern_layout;                                   \
    using OutputLayout = _output_layout;                               \
    using Param = _conv_param;                                         \
    static constexpr bool check_bounds = check_bounds_;
#define MEGDNN_COMMA ,

template <bool check_bounds_, typename src_ldg_dtype, typename filter_ldg_dtype,
          typename RegBlockConfig_, typename ThreadConfig_>
struct IBatchConvTrait_f1x1s1x1 {
    COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM(int8_t, int8_t, int32_t,
                                                Layout<Format::NCHW4>,
                                                Layout<Format::NCHW4>,
                                                Layout<Format::NCHW4>,
                                                ConvParam);
    using RegBlockConfig = RegBlockConfig_;
    using ThreadConfig = ThreadConfig_;
    struct DataTileCount {
        using RegBlockConfig = RegBlockConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = src_ldg_dtype;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(src_dtype);
        static int constexpr skew = load_width;
        static int constexpr block_tile_batch = RegBlockConfig::reg_n;
        MEGDNN_STATIC_ASSERT(
                block_tile_batch == 1,
                "this algorithm does not unroll on batch dimension");
        static int constexpr block_tile_out_height_width =
                RegBlockConfig::reg_width * ThreadConfig::nr_thread_x;
        static int constexpr block_tile_in_channel = RegBlockConfig::reg_k;

        static int constexpr smem_load_x =
                block_tile_out_height_width / load_width;
        static int constexpr load_x =
                smem_load_x > WARP_SIZE ? WARP_SIZE : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        static int constexpr smem_h = RegBlockConfig::reg_k_packed;
        static int constexpr smem_w = block_tile_out_height_width;
        static int constexpr smem_stride =
                smem_w % 2 == 0 ? smem_w + skew : smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    struct FilterTileCount {
        using RegBlockConfig = RegBlockConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = filter_ldg_dtype;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(filter_dtype);
        static int constexpr skew = load_width;
        static int constexpr block_tile_out_channel =
                RegBlockConfig::reg_m * ThreadConfig::nr_thread_y;
        static int constexpr block_tile_in_channel = RegBlockConfig::reg_k;

        static int constexpr smem_load_x =
                RegBlockConfig::reg_k_packed / load_width;
        static int constexpr load_x =
                smem_load_x > WARP_SIZE ? WARP_SIZE : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        static int constexpr smem_h = block_tile_out_channel;
        static int constexpr smem_w = RegBlockConfig::reg_k_packed;
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    using BlockTileIterator =
            BlockTileIterator_COxHW<DataTileCount, FilterTileCount,
                                    BatchConvPrologue>;
    using DataGlobal2ShareMemVisitor =
            Global2ShareMemVisitor_CIxHW<check_bounds, false, DataTileCount,
                                         InputLayout>;
    using FilterGlobal2ShareMemVisitor =
            Global2ShareMemVisitor_COxCI<check_bounds, FilterTileCount,
                                         KernLayout>;
    static bool constexpr pipelined = RegBlockConfig::reg_k_packed > 1;
    using BlockConsumer =
            IConvBlockConsumer_COxHW<RegBlockConfig, ThreadConfig, pipelined>;
    using GlobalMemoryWriter =
            IConvGlobalMemoryWriter_COxHW<RegBlockConfig, ThreadConfig>;
};

template <bool check_bounds_, typename filter_ldg_dtype,
          typename RegBlockConfig_, typename ThreadConfig_>
struct IBatchConvTrait {
    COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM(int8_t, int8_t, int32_t,
                                                Layout<Format::NCHW4>,
                                                Layout<Format::NCHW4>,
                                                Layout<Format::NCHW4>,
                                                ConvParam);
    using RegBlockConfig = RegBlockConfig_;
    using ThreadConfig = ThreadConfig_;
    struct DataTileCount {
        using RegBlockConfig = RegBlockConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = int32_t;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width = 4;
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(src_dtype);
        static int constexpr skew = load_width;
        static int constexpr block_tile_batch = RegBlockConfig::reg_n;
        MEGDNN_STATIC_ASSERT(
                block_tile_batch == 1,
                "this algorithm does not unroll on batch dimension");
        static int constexpr block_tile_out_height_width =
                RegBlockConfig::reg_width * ThreadConfig::nr_thread_x;
        static int constexpr block_tile_in_channel = RegBlockConfig::reg_k;

        static int constexpr smem_load_x =
                DIVUP(block_tile_out_height_width, load_width);
        static int constexpr load_x =
                smem_load_x > WARP_SIZE ? WARP_SIZE : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        static int constexpr smem_h = RegBlockConfig::reg_k_packed;
        static int constexpr smem_w = smem_load_x * load_width;
        static int constexpr smem_stride =
                smem_w % 2 == 0 ? smem_w + skew : smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;
        static int constexpr reg_d = load_width;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };
    using FilterTileCount =
            typename IBatchConvTrait_f1x1s1x1<check_bounds, int,
                                              filter_ldg_dtype, RegBlockConfig,
                                              ThreadConfig>::FilterTileCount;

    using BlockTileIterator =
            BlockTileIterator_COxHW<DataTileCount, FilterTileCount,
                                    BatchConvPrologue>;
    using DataGlobal2ShareMemVisitor =
            Global2ShareMemVisitor_CIxHW<check_bounds, true, DataTileCount,
                                         InputLayout>;
    using FilterGlobal2ShareMemVisitor =
            Global2ShareMemVisitor_COxCI<check_bounds, FilterTileCount,
                                         KernLayout>;
    static bool constexpr pipelined = RegBlockConfig::reg_k_packed > 1;
    using BlockConsumer =
            IConvBlockConsumer_COxHW<RegBlockConfig, ThreadConfig, pipelined>;
    using GlobalMemoryWriter =
            IConvGlobalMemoryWriter_COxHW<RegBlockConfig, ThreadConfig>;
};

#undef COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM
#undef MEGDNN_COMMA

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
