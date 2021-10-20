/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 *permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this
 *list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this
 *list of conditions and the following disclaimer in the documentation and/or other
 *materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors
 *may be used to endorse or promote products derived from this software without specific
 *prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 *EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 *OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 *SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 *HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file
 * dnn/src/cuda/convolution_helper/global_memory_writer/iconv_global_memory_writer_coxhw.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

namespace megdnn {
namespace cuda {
namespace convolution {
template <typename RegBlockConfig_, typename ThreadConfig_>
struct IConvGlobalMemoryWriter_COxHW {
    using RegBlockConfig = RegBlockConfig_;
    using ThreadConfig = ThreadConfig_;

    float alpha;
    float beta;
    int block_out_height_width_remain;
    int block_out_channel_remain;

    __device__ __forceinline__ void init(
            int32_t* /* smem */, const float alpha_, const float beta_) {
        alpha = alpha_, beta = beta_;
    }

    template <
            bool check_bounds, typename BiasVisitor, typename Epilogue,
            typename BlockConsumer>
    __device__ __forceinline__ void write(
            BiasVisitor bias, Epilogue epilogue, BlockConsumer block_consumer) {
        static constexpr bool use_wide_store = !(RegBlockConfig::reg_width & 0x1);
        static constexpr int pack_size_bit = RegBlockConfig::pack_size_bit;

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        if (use_wide_store) {
#pragma unroll
            for (int i = 0; i < RegBlockConfig::reg_m_packed; ++i) {
#pragma unroll
                for (int j = 0; j < (RegBlockConfig::reg_width >> 1); ++j) {
                    int j2 = (j << 1);
                    int out_channel =
                            ((tidy + i * ThreadConfig::nr_thread_y) << pack_size_bit);
                    int out_height_width = (tidx << 1) + j2 * ThreadConfig::nr_thread_x;
                    int ipack = (i << pack_size_bit);
                    float4 f_conv0 = make_float4(
                            block_consumer.reg_acc[j2][ipack],
                            block_consumer.reg_acc[j2][ipack + 1],
                            block_consumer.reg_acc[j2][ipack + 2],
                            block_consumer.reg_acc[j2][ipack + 3]);
                    float4 f_conv1 = make_float4(
                            block_consumer.reg_acc[j2 + 1][ipack],
                            block_consumer.reg_acc[j2 + 1][ipack + 1],
                            block_consumer.reg_acc[j2 + 1][ipack + 2],
                            block_consumer.reg_acc[j2 + 1][ipack + 3]);
                    //                    if (threadIdx.x == 0 && threadIdx.y == 0 &&
                    //                    blockIdx.x == 0 && blockIdx.y == 0 &&
                    //                    blockIdx.z == 0 && i == 0 && j == 0) {
                    //                        printf("acc = %f, %f, %f, %f\n",
                    //                        f_conv0.x, f_conv0.y, f_conv0.z,
                    //                        f_conv0.w);
                    //                    }

                    if (!check_bounds) {
                        float4 f_bias0 = bias.at(0, out_channel, out_height_width);
                        float4 f_bias1 = bias.at(0, out_channel, out_height_width + 1);
                        epilogue.apply(
                                alpha, f_conv0, f_conv1, beta, f_bias0, f_bias1, 0,
                                out_channel, out_height_width);
                    } else if (out_channel < block_out_channel_remain) {
                        if (((block_out_height_width_remain & 0x1) == 0) &&
                            out_height_width + 2 <= block_out_height_width_remain) {
                            float4 f_bias0 = bias.at(0, out_channel, out_height_width);
                            float4 f_bias1 =
                                    bias.at(0, out_channel, out_height_width + 1);
                            epilogue.apply(
                                    alpha, f_conv0, f_conv1, beta, f_bias0, f_bias1, 0,
                                    out_channel, out_height_width);
                        } else {
#define store(_i)                                                      \
    if (out_height_width + (_i) < block_out_height_width_remain) {     \
        float4 f_bias##_i = bias.at(0, out_channel, out_height_width); \
        epilogue.apply(                                                \
                alpha, f_conv##_i, beta, f_bias##_i, 0, out_channel,   \
                out_height_width + (_i));                              \
    }
                            store(0);
                            store(1);
#undef store
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < RegBlockConfig::reg_m_packed; ++i) {
#pragma unroll
                for (int j = 0; j < RegBlockConfig::reg_width; ++j) {
                    int out_channel =
                            ((tidy + i * ThreadConfig::nr_thread_y) << pack_size_bit);
                    int out_height_width = tidx + j * ThreadConfig::nr_thread_x;
                    int ipack = (i << pack_size_bit);
                    if (check_bounds &&
                        (out_channel >= block_out_channel_remain ||
                         out_height_width >= block_out_height_width_remain)) {
                    } else {
                        float4 f_conv = make_float4(
                                block_consumer.reg_acc[j][ipack],
                                block_consumer.reg_acc[j][ipack + 1],
                                block_consumer.reg_acc[j][ipack + 2],
                                block_consumer.reg_acc[j][ipack + 3]);
                        float4 f_bias = bias.at(0, out_channel, out_height_width);
                        epilogue.apply(
                                alpha, f_conv, beta, f_bias, 0, out_channel,
                                out_height_width);
                    }
                }
            }
        }
    }
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
