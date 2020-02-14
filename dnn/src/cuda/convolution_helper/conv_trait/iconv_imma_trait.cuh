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
 * \file dnn/src/cuda/convolution_helper/conv_trait/iconv_imma_trait.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/block_tile_consumer/block_consumer.cuh"
#include "src/cuda/convolution_helper/block_tile_iterator/block_tile_iterator.cuh"
#include "src/cuda/convolution_helper/global_memory_visitor/global_memory_visitor.cuh"
#include "src/cuda/convolution_helper/global_memory_writer/global_memory_writer.cuh"
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

template <bool check_bounds_, typename IMMAConfig_, typename WarpTileConfig_,
          typename ThreadConfig_>
struct IConvIMMATrait {
    COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM(int8_t, int8_t, int32_t,
                                                Layout<Format::CHWN4>,
                                                Layout<Format::CHWN4>,
                                                Layout<Format::CHWN4>,
                                                ConvParam);
    using IMMAConfig = IMMAConfig_;
    using WarpTileConfig = WarpTileConfig_;
    using ThreadConfig = ThreadConfig_;
    struct DataTileCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = int32_t;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(src_dtype);
        static int constexpr block_tile_batch = WarpTileConfig::warp_tile_n *
                                                IMMAConfig::wmma_n *
                                                ThreadConfig::nr_warp_x;
        static int constexpr block_tile_in_channel =
                WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k;

        static int constexpr smem_load_x = block_tile_batch / load_width;
        static int constexpr load_x = smem_load_x > 32 ? 32 : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        // smem col major
        static int constexpr smem_h = WarpTileConfig::warp_tile_k;
        static int constexpr smem_w = IMMAConfig::tile_b_sizes_int *
                                      WarpTileConfig::warp_tile_n *
                                      ThreadConfig::nr_warp_x;
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;
        static int constexpr reg_d =
                IMMAConfig::wmma_k / WarpTileConfig::pack_size;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    struct FilterTileCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = int32_t;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(filter_dtype);
        static int constexpr block_tile_out_channel =
                WarpTileConfig::warp_tile_m * IMMAConfig::wmma_m *
                ThreadConfig::nr_warp_y;
        static int constexpr block_tile_in_channel =
                WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k;

        static int constexpr smem_load_x = block_tile_out_channel / load_width;
        static int constexpr load_x = smem_load_x > 32 ? 32 : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        // smem col major
        static int constexpr smem_h = WarpTileConfig::warp_tile_k;
        static int constexpr smem_w = IMMAConfig::tile_a_sizes_int *
                                      WarpTileConfig::warp_tile_m *
                                      ThreadConfig::nr_warp_y;
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;
        static int constexpr reg_d =
                IMMAConfig::wmma_k / WarpTileConfig::pack_size;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    struct GlobalMemoryStoreCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = int4;
        static int constexpr smem_h = ThreadConfig::nr_warp_y;
        static int constexpr smem_w =
                (WarpTileConfig::warp_tile_n & 0x1)
                        ? ThreadConfig::nr_warp_x * IMMAConfig::wmma_m *
                                  IMMAConfig::wmma_n
                        : 2 * ThreadConfig::nr_warp_x * IMMAConfig::wmma_m *
                                  IMMAConfig::wmma_n;
        static int constexpr store_width = sizeof(copy_t) / sizeof(int32_t);
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr store_x =
                (WarpTileConfig::warp_tile_n & 0x1)
                        ? IMMAConfig::wmma_n / store_width
                        : 2 * IMMAConfig::wmma_n / store_width;
        static int constexpr store_y = ThreadConfig::warp_size / store_x;
    };

    using BlockTileIterator =
            BlockTileIteratorBasic<DataTileCount, FilterTileCount>;
    using DataGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_CIxN<check_bounds, DataTileCount,
                                            InputLayout>;
    using FilterGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_CIxN<check_bounds, FilterTileCount,
                                            KernLayout>;
    static bool constexpr pipelined = WarpTileConfig::warp_tile_k > 1;
    using BlockConsumer = IConvIMMABlockConsumer<IMMAConfig, WarpTileConfig,
                                                 ThreadConfig, pipelined>;
    using GlobalMemoryWriter =
            IConvIMMAGlobalMemoryWriter<GlobalMemoryStoreCount>;
};

template <bool check_bounds_, typename IMMAConfig_, typename WarpTileConfig_,
          typename ThreadConfig_>
struct IConvIMMATraitReorderFilter {
    COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM(int8_t, int8_t, int32_t,
                                                Layout<Format::CHWN4>,
                                                Layout<Format::CHWN16>,
                                                Layout<Format::CHWN4>,
                                                ConvParam);
    using IMMAConfig = IMMAConfig_;
    using WarpTileConfig = WarpTileConfig_;
    using ThreadConfig = ThreadConfig_;
    MEGDNN_STATIC_ASSERT(
            std::is_same<typename IConvIMMATrait<
                            check_bounds MEGDNN_COMMA IMMAConfig MEGDNN_COMMA
                                    WarpTileConfig MEGDNN_COMMA ThreadConfig>::
                                 src_dtype MEGDNN_COMMA src_dtype>::value ==
                    true,
            "data type of input tensor should be int8_t");
    using DataTileCount =
            typename IConvIMMATrait<check_bounds, IMMAConfig, WarpTileConfig,
                                    ThreadConfig>::DataTileCount;
    struct FilterTileCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = int4;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(filter_dtype);
        static int constexpr block_tile_out_channel =
                WarpTileConfig::warp_tile_m * IMMAConfig::wmma_m *
                ThreadConfig::nr_warp_y;
        static int constexpr block_tile_in_channel =
                WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k;

        static int constexpr smem_load_x = block_tile_out_channel;
        static int constexpr load_x = smem_load_x > 32 ? 32 : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        // smem col major
        static int constexpr smem_h = WarpTileConfig::warp_tile_k;
        static int constexpr smem_w = IMMAConfig::tile_a_sizes_int *
                                      WarpTileConfig::warp_tile_m *
                                      ThreadConfig::nr_warp_y;
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    using BlockTileIterator =
            BlockTileIteratorBasic<DataTileCount, FilterTileCount>;
    using DataGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_CIxN<check_bounds, DataTileCount,
                                            InputLayout>;
    using FilterGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_CIxN<check_bounds, FilterTileCount,
                                            KernLayout>;
    static bool constexpr pipelined = WarpTileConfig::warp_tile_k > 1;
    using BlockConsumer = IConvIMMABlockConsumer<IMMAConfig, WarpTileConfig,
                                                 ThreadConfig, pipelined>;
    using GlobalMemoryStoreCount =
            typename IConvIMMATrait<check_bounds, IMMAConfig, WarpTileConfig,
                                    ThreadConfig>::GlobalMemoryStoreCount;
    using GlobalMemoryWriter =
            IConvIMMAGlobalMemoryWriter<GlobalMemoryStoreCount>;
};

template <bool check_bounds_, typename IMMAConfig_, typename WarpTileConfig_,
          typename ThreadConfig_>
struct IConvIMMATraitUnrollWidth {
    COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM(int8_t, int8_t, int32_t,
                                                Layout<Format::CHWN4>,
                                                Layout<Format::CHWN16>,
                                                Layout<Format::CHWN4>,
                                                ConvParam);
    using IMMAConfig = IMMAConfig_;
    using WarpTileConfig = WarpTileConfig_;
    using ThreadConfig = ThreadConfig_;

    struct DataTileCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = int4;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(src_dtype);

        static int constexpr block_tile_batch = IMMAConfig::wmma_n;
        static int constexpr block_tile_out_width =
                WarpTileConfig::warp_tile_n * ThreadConfig::nr_warp_x;
        static int constexpr block_tile_in_channel =
                WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k;

        static int constexpr smem_load_x =
                block_tile_batch * block_tile_out_width / load_width;
        static int constexpr load_x = smem_load_x > 32 ? 32 : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        // smem col major
        static int constexpr smem_h = WarpTileConfig::warp_tile_k;
        static int constexpr smem_w =
                IMMAConfig::tile_b_sizes_int * block_tile_out_width;
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;
        static int constexpr reg_d =
                IMMAConfig::wmma_k / WarpTileConfig::pack_size;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    MEGDNN_STATIC_ASSERT(
            std::is_same<typename IConvIMMATraitReorderFilter<
                            check_bounds MEGDNN_COMMA IMMAConfig MEGDNN_COMMA
                                    WarpTileConfig MEGDNN_COMMA
                                            ThreadConfig>::filter_dtype
                                 MEGDNN_COMMA filter_dtype>::value == true,
            "data type of filter tensor should be int8_t");
    using FilterTileCount =
            typename IConvIMMATraitReorderFilter<check_bounds, IMMAConfig,
                                                 WarpTileConfig,
                                                 ThreadConfig>::FilterTileCount;
    using BlockTileIterator =
            BlockTileIteratorUnrollWidth<DataTileCount, FilterTileCount>;
    using DataGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_CIxWOxN<check_bounds, DataTileCount,
                                               InputLayout>;
    using FilterGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_CIxN<check_bounds, FilterTileCount,
                                            KernLayout>;
    static bool constexpr pipelined = WarpTileConfig::warp_tile_k > 1;
    using BlockConsumer = IConvIMMABlockConsumer<IMMAConfig, WarpTileConfig,
                                                 ThreadConfig, pipelined>;

    struct GlobalMemoryStoreCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using copy_t = int4;
        static int constexpr smem_h = ThreadConfig::nr_warp_y;
        static int constexpr consecutive_width_tile =
                !(WarpTileConfig::warp_tile_n & 0x1);
        static int constexpr smem_w =
                consecutive_width_tile
                        ? 2 * ThreadConfig::nr_warp_x * IMMAConfig::wmma_m *
                                  IMMAConfig::wmma_n
                        : ThreadConfig::nr_warp_x * IMMAConfig::wmma_m *
                                  IMMAConfig::wmma_n;

        static int constexpr store_width = sizeof(copy_t) / sizeof(int32_t);
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr store_x =
                consecutive_width_tile ? 2 * IMMAConfig::wmma_n / store_width
                                       : IMMAConfig::wmma_n / store_width;
        static int constexpr store_y = ThreadConfig::warp_size / store_x;
    };
    using GlobalMemoryWriter =
            IConvIMMAGlobalMemoryWriterUnrollWidth<GlobalMemoryStoreCount>;
};

template <bool check_bounds_, typename Conv1dConfig_, typename IMMAConfig_,
          typename WarpTileConfig_, typename ThreadConfig_>
struct IConvIMMATraitUnrollWidthV2 {
    COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM(int8_t, int8_t, int32_t,
                                                Layout<Format::CHWN4>,
                                                Layout<Format::CHWN16>,
                                                Layout<Format::CHWN4>,
                                                ConvParam);
    using Conv1dConfig = Conv1dConfig_;
    using IMMAConfig = IMMAConfig_;
    using WarpTileConfig = WarpTileConfig_;
    using ThreadConfig = ThreadConfig_;

    struct DataTileCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using Conv1dConfig = Conv1dConfig;

        MEGDNN_STATIC_ASSERT(WarpTileConfig::warp_tile_k == 1,
                             "kernel unrolling along width axis assumes tile k "
                             "in warp-level must be 1");
        using copy_t = int4;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(src_dtype);

        static int constexpr block_tile_out_width =
                WarpTileConfig::warp_tile_n * ThreadConfig::nr_warp_x;
        static int constexpr block_tile_in_width =
                (WarpTileConfig::warp_tile_n * ThreadConfig::nr_warp_x - 1) *
                        Conv1dConfig::sw +
                Conv1dConfig::fw;
        static int constexpr block_tile_batch = IMMAConfig::wmma_n;
        static int constexpr block_tile_in_channel =
                WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k;

        static int constexpr smem_load_x = block_tile_batch / load_width;
        static int constexpr load_x = smem_load_x > 32 ? 32 : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        // smem col major
        static int constexpr smem_h =
                WarpTileConfig::warp_tile_k * block_tile_in_width;
        static int constexpr smem_w = IMMAConfig::tile_b_sizes_int;
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;
        static int constexpr reg_d =
                IMMAConfig::wmma_k / WarpTileConfig::pack_size;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    struct FilterTileCount {
        using IMMAConfig = IMMAConfig;
        using WarpTileConfig = WarpTileConfig;
        using ThreadConfig = ThreadConfig;
        using Conv1dConfig = Conv1dConfig;

        MEGDNN_STATIC_ASSERT(WarpTileConfig::warp_tile_k == 1,
                             "kernel unrolling along width axis assumes tile k "
                             "in warp-level must be 1");
        using copy_t = int4;
        using smem_storage_dtype = smem_storage_dtype;
        static int constexpr load_width =
                sizeof(copy_t) / sizeof(smem_storage_dtype);
        static int constexpr ldg_load_width =
                sizeof(copy_t) / sizeof(filter_dtype);
        static int constexpr block_tile_out_channel =
                WarpTileConfig::warp_tile_m * IMMAConfig::wmma_m *
                ThreadConfig::nr_warp_y;
        static int constexpr block_tile_in_channel =
                WarpTileConfig::warp_tile_k * IMMAConfig::wmma_k;

        static int constexpr smem_load_x = block_tile_out_channel;
        static int constexpr load_x = smem_load_x > 32 ? 32 : smem_load_x;
        static int constexpr load_y = ThreadConfig::nr_threads / load_x;

        // smem col major
        static int constexpr smem_h = Conv1dConfig::fw;
        static int constexpr smem_w = IMMAConfig::tile_a_sizes_int *
                                      WarpTileConfig::warp_tile_m *
                                      ThreadConfig::nr_warp_y;
        static int constexpr smem_stride = smem_w;
        static int constexpr smem_tot = smem_h * smem_stride;

        static int constexpr reg_h = (smem_h + load_y - 1) / load_y;
        static int constexpr reg_w = (smem_load_x + load_x - 1) / load_x;

        static bool constexpr check_bounds_h = smem_h % load_y != 0;
        static bool constexpr check_bounds_w = smem_load_x % load_x != 0;
    };

    using BlockTileIterator =
            BlockTileIteratorUnrollWidthV2<DataTileCount, FilterTileCount>;
    using DataGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_CIxWIxN<check_bounds, DataTileCount,
                                               InputLayout>;
    using FilterGlobal2ShareMemVisitor =
            Global2ShareMemVisitorIMMA_FWxCO<check_bounds, FilterTileCount,
                                             KernLayout>;
    using BlockConsumer =
            IConvIMMABlockConsumerUnrollWidth<Conv1dConfig, IMMAConfig,
                                              WarpTileConfig, ThreadConfig>;
    using GlobalMemoryStoreCount = typename IConvIMMATraitUnrollWidth<
            check_bounds, IMMAConfig, WarpTileConfig,
            ThreadConfig>::GlobalMemoryStoreCount;
    using GlobalMemoryWriter =
            IConvIMMAGlobalMemoryWriterUnrollWidth<GlobalMemoryStoreCount>;
};
#undef COMMON_DEFS_WITH_DATA_TYPE_LAYOUT_AND_PARAM
#undef MEGDNN_COMMA

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
