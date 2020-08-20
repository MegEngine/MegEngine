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
 * \file dnn/src/cuda/convolution_helper/kernel.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/bias_visitor.cuh"
#include "src/cuda/convolution_helper/config.cuh"
#include "src/cuda/convolution_helper/conv_trait/conv_trait.cuh"
#include "src/cuda/convolution_helper/epilogue.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {

template <typename ConvTrait, typename BiasVisitor, typename Epilogue>
__global__ void convolution_kernel(
        const typename ConvTrait::src_dtype* __restrict__ src,
        const typename ConvTrait::filter_dtype* __restrict__ filter,
        BiasVisitor bias, Epilogue epilogue, typename ConvTrait::Param param,
        float alpha, float beta) {
    static bool constexpr check_bounds = ConvTrait::check_bounds;
    using BlockTileIterator = typename ConvTrait::BlockTileIterator;
    BlockTileIterator block_iterator;
    // determine batch, out_channel, out_height, out_width of current thread
    // block
    block_iterator.template init_with_param<ConvTrait::Param>(param);

    using DataTileCount = typename ConvTrait::DataTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;

    using DataGlobal2ShareMemVisitor =
            typename ConvTrait::DataGlobal2ShareMemVisitor;
    using FilterGlobal2ShareMemVisitor =
            typename ConvTrait::FilterGlobal2ShareMemVisitor;

    using smem_storage_dtype = typename ConvTrait::smem_storage_dtype;
    extern __shared__ smem_storage_dtype smem[];
    smem_storage_dtype* smem_src = smem;
    smem_storage_dtype* smem_filter = smem + DataTileCount::smem_tot;
    smem_storage_dtype* smem_dst = smem_filter + FilterTileCount::smem_tot;

    DataGlobal2ShareMemVisitor src_gl2sh_visitor{smem_src};
    FilterGlobal2ShareMemVisitor filter_gl2sh_visitor{smem_filter};
    if (check_bounds) {
        block_iterator.set_remain(src_gl2sh_visitor,
                                           filter_gl2sh_visitor);
    }

    using BlockConsumer = typename ConvTrait::BlockConsumer;
    BlockConsumer block_consumer;
    block_consumer.init_accumulator();

    block_iterator.template iterate_with_param<typename ConvTrait::InputLayout,
                                               typename ConvTrait::KernLayout>(
            src, filter, param, src_gl2sh_visitor, filter_gl2sh_visitor,
            block_consumer);

    using GlobalMemoryWriter = typename ConvTrait::GlobalMemoryWriter;
    GlobalMemoryWriter global_memory_writer;
    global_memory_writer.init(smem_dst, alpha, beta);
    if (check_bounds) {
        block_iterator.set_remain(global_memory_writer);
    }
    bias.move(block_iterator.block_batch, block_iterator.block_out_channel,
              block_iterator.block_out_height, block_iterator.block_out_width);
    epilogue.move(block_iterator.block_batch, block_iterator.block_out_channel,
                  block_iterator.block_out_height,
                  block_iterator.block_out_width);
    global_memory_writer.template write<check_bounds>(bias, epilogue,
                                                      block_consumer);
}

template <typename ConvTrait, typename BiasVisitor, typename Epilogue>
__global__ void convolution_kernel_precomp_offset(
        const typename ConvTrait::src_dtype* __restrict__ src,
        const typename ConvTrait::filter_dtype* __restrict__ filter,
        const int* __restrict__ offset, BiasVisitor bias, Epilogue epilogue,
        typename ConvTrait::Param param, float alpha, float beta) {
    static bool constexpr check_bounds = ConvTrait::check_bounds;
    using BlockTileIterator = typename ConvTrait::BlockTileIterator;
    BlockTileIterator block_iterator;
    // determine batch, out_channel, out_height, out_width of current thread
    // block
    block_iterator.template init_with_param<ConvTrait::Param>(param);

    using DataTileCount = typename ConvTrait::DataTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;

    using DataGlobal2ShareMemVisitor =
            typename ConvTrait::DataGlobal2ShareMemVisitor;
    using FilterGlobal2ShareMemVisitor =
            typename ConvTrait::FilterGlobal2ShareMemVisitor;

    using smem_storage_dtype = typename ConvTrait::smem_storage_dtype;
    extern __shared__ smem_storage_dtype smem[];
    smem_storage_dtype* smem_src = smem;
    smem_storage_dtype* smem_filter = smem + DataTileCount::smem_tot;
    smem_storage_dtype* smem_dst = smem_filter + FilterTileCount::smem_tot;

    DataGlobal2ShareMemVisitor src_gl2sh_visitor{smem_src, offset};
    FilterGlobal2ShareMemVisitor filter_gl2sh_visitor{smem_filter};
    if (check_bounds) {
        block_iterator.set_remain(src_gl2sh_visitor,
                                           filter_gl2sh_visitor);
    }

    using BlockConsumer = typename ConvTrait::BlockConsumer;
    BlockConsumer block_consumer;
    block_consumer.init_accumulator();

    block_iterator.template iterate_with_param<typename ConvTrait::InputLayout,
                                               typename ConvTrait::KernLayout>(
            src, filter, param, src_gl2sh_visitor, filter_gl2sh_visitor,
            block_consumer);

    using GlobalMemoryWriter = typename ConvTrait::GlobalMemoryWriter;
    GlobalMemoryWriter global_memory_writer;
    global_memory_writer.init(smem_dst, alpha, beta);
    if (check_bounds) {
        block_iterator.set_remain(global_memory_writer);
    }
    bias.move(block_iterator.block_batch, block_iterator.block_out_channel,
              block_iterator.block_out_height, block_iterator.block_out_width);
    epilogue.move(block_iterator.block_batch, block_iterator.block_out_channel,
                  block_iterator.block_out_height,
                  block_iterator.block_out_width);
    global_memory_writer.template write<check_bounds>(bias, epilogue,
                                                      block_consumer);
}

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
