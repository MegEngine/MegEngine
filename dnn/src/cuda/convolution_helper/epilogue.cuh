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
 * \file dnn/src/cuda/convolution_helper/epilogue.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/activation.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {

template <typename ActivationOp>
struct IConvEpilogue {
    int8_t* __restrict__ dst;
    const int8_t* __restrict__ z;
    int batch_stride;
    int channel_stride;
    int height_stride;
    int width_stride;
    float gamma;
    ActivationOp act;
    MEGDNN_HOST MEGDNN_DEVICE IConvEpilogue(int8_t* __restrict__ dst,
                                            const int8_t* __restrict__ z,
                                            int batch_stride,
                                            int channel_stride,
                                            int height_stride, int width_stride,
                                            float gamma, ActivationOp act)
            : dst{dst},
              z{z},
              batch_stride{batch_stride},
              channel_stride{channel_stride},
              height_stride{height_stride},
              width_stride{width_stride},
              gamma{gamma},
              act{act} {}
#if MEGDNN_CC_CUDA
    __device__ __forceinline__ void move(const int b_idx, const int ch_idx,
                                         const int h_idx, const int w_idx) {
        size_t offset = b_idx * batch_stride + ch_idx * channel_stride +
                        h_idx * height_stride + w_idx * width_stride;
        dst += offset;
        if (z != nullptr)
            z += offset;
    }
    __device__ __forceinline__ void apply(float alpha, float4 f_conv,
                                          float beta, float4 f_bias,
                                          const int b_idx, const int ch_idx,
                                          const int h_idx, const int w_idx) {
        size_t idx = b_idx * batch_stride + ch_idx * channel_stride +
                     h_idx * height_stride + w_idx * width_stride;
        float4 f_res = alpha * f_conv + beta * f_bias;
        if (z != nullptr) {
            int i_z = __ldg(reinterpret_cast<const int32_t*>(&z[idx]));
            float4 f_z = transform_int8x4_to_float4(i_z);
            f_res = f_res + gamma * f_z;
        }
        *(reinterpret_cast<int32_t*>(&dst[idx])) =
                act.apply_and_transform(f_res);
    }
    __device__ __forceinline__ void apply(float alpha, float4 f_conv,
                                          float beta, float4 f_bias,
                                          const int b_idx, const int ch_idx,
                                          const int hw_idx) {
        size_t idx = b_idx * batch_stride + ch_idx * channel_stride +
                     hw_idx * width_stride;
        float4 f_res = alpha * f_conv + beta * f_bias;
        if (z != nullptr) {
            int i_z = __ldg(reinterpret_cast<const int32_t*>(&z[idx]));
            float4 f_z = transform_int8x4_to_float4(i_z);
            f_res = f_res + gamma * f_z;
        }
        *(reinterpret_cast<int32_t*>(&dst[idx])) =
                act.apply_and_transform(f_res);
    }
    __device__ __forceinline__ void apply(float alpha, float4 f_conv_x,
                                          float4 f_conv_y, float beta,
                                          float4 f_bias_x, float4 f_bias_y,
                                          const int b_idx, const int ch_idx,
                                          const int h_idx, const int w_idx) {
        size_t idx = b_idx * batch_stride + ch_idx * channel_stride +
                     h_idx * height_stride + w_idx * width_stride;
        float4 f_res_x = alpha * f_conv_x + beta * f_bias_x;
        float4 f_res_y = alpha * f_conv_y + beta * f_bias_y;
        if (z != nullptr) {
            int2 i_z2 = __ldg(reinterpret_cast<const int2*>(&z[idx]));
            float4 f_z_x = transform_int8x4_to_float4(i_z2.x);
            float4 f_z_y = transform_int8x4_to_float4(i_z2.y);
            f_res_x = f_res_x + gamma * f_z_x;
            f_res_y = f_res_y + gamma * f_z_y;
        }
        int ix = act.apply_and_transform(f_res_x);
        int iy = act.apply_and_transform(f_res_y);
        *(reinterpret_cast<int2*>(&dst[idx])) = ::make_int2(ix, iy);
    }
    __device__ __forceinline__ void apply(float alpha, float4 f_conv_x,
                                          float4 f_conv_y, float beta,
                                          float4 f_bias_x, float4 f_bias_y,
                                          const int b_idx, const int ch_idx,
                                          const int hw_idx) {
        size_t idx = b_idx * batch_stride + ch_idx * channel_stride +
                     hw_idx * width_stride;
        float4 f_res_x = alpha * f_conv_x + beta * f_bias_x;
        float4 f_res_y = alpha * f_conv_y + beta * f_bias_y;
        if (z != nullptr) {
            int2 i_z2 = __ldg(reinterpret_cast<const int2*>(&z[idx]));
            float4 f_z_x = transform_int8x4_to_float4(i_z2.x);
            float4 f_z_y = transform_int8x4_to_float4(i_z2.y);
            f_res_x = f_res_x + gamma * f_z_x;
            f_res_y = f_res_y + gamma * f_z_y;
        }
        int ix = act.apply_and_transform(f_res_x);
        int iy = act.apply_and_transform(f_res_y);
        *(reinterpret_cast<int2*>(&dst[idx])) = ::make_int2(ix, iy);
    }

    __device__ __forceinline__ void apply(float alpha, float4 f_conv_x,
                                          float4 f_conv_y, float4 f_conv_z,
                                          float4 f_conv_w, float beta,
                                          float4 f_bias_x, float4 f_bias_y,
                                          float4 f_bias_z, float4 f_bias_w,
                                          const int b_idx, const int ch_idx,
                                          const int h_idx, const int w_idx) {
        size_t idx = b_idx * batch_stride + ch_idx * channel_stride +
                     h_idx * height_stride + w_idx * width_stride;
        float4 f_res_x = alpha * f_conv_x + beta * f_bias_x;
        float4 f_res_y = alpha * f_conv_y + beta * f_bias_y;
        float4 f_res_z = alpha * f_conv_z + beta * f_bias_z;
        float4 f_res_w = alpha * f_conv_w + beta * f_bias_w;
        if (z != nullptr) {
            int4 i_z4 = __ldg(reinterpret_cast<const int4*>(&z[idx]));

            float4 f_z_x = transform_int8x4_to_float4(i_z4.x);
            float4 f_z_y = transform_int8x4_to_float4(i_z4.y);
            float4 f_z_z = transform_int8x4_to_float4(i_z4.z);
            float4 f_z_w = transform_int8x4_to_float4(i_z4.w);

            f_res_x = f_res_x + gamma * f_z_x;
            f_res_y = f_res_y + gamma * f_z_y;
            f_res_z = f_res_z + gamma * f_z_z;
            f_res_w = f_res_w + gamma * f_z_w;
        }
        int ix = act.apply_and_transform(f_res_x);
        int iy = act.apply_and_transform(f_res_y);
        int iz = act.apply_and_transform(f_res_z);
        int iw = act.apply_and_transform(f_res_w);
        *(reinterpret_cast<int4*>(&dst[idx])) = ::make_int4(ix, iy, iz, iw);
    }
    __device__ __forceinline__ void apply(float alpha, float4 f_conv_x,
                                          float4 f_conv_y, float4 f_conv_z,
                                          float4 f_conv_w, float beta,
                                          float4 f_bias_x, float4 f_bias_y,
                                          float4 f_bias_z, float4 f_bias_w,
                                          const int b_idx, const int ch_idx,
                                          const int hw_idx) {
        size_t idx = b_idx * batch_stride + ch_idx * channel_stride +
                     hw_idx * width_stride;
        float4 f_res_x = alpha * f_conv_x + beta * f_bias_x;
        float4 f_res_y = alpha * f_conv_y + beta * f_bias_y;
        float4 f_res_z = alpha * f_conv_z + beta * f_bias_z;
        float4 f_res_w = alpha * f_conv_w + beta * f_bias_w;
        if (z != nullptr) {
            int4 i_z4 = __ldg(reinterpret_cast<const int4*>(&z[idx]));

            float4 f_z_x = transform_int8x4_to_float4(i_z4.x);
            float4 f_z_y = transform_int8x4_to_float4(i_z4.y);
            float4 f_z_z = transform_int8x4_to_float4(i_z4.z);
            float4 f_z_w = transform_int8x4_to_float4(i_z4.w);

            f_res_x = f_res_x + gamma * f_z_x;
            f_res_y = f_res_y + gamma * f_z_y;
            f_res_z = f_res_z + gamma * f_z_z;
            f_res_w = f_res_w + gamma * f_z_w;
        }
        int ix = act.apply_and_transform(f_res_x);
        int iy = act.apply_and_transform(f_res_y);
        int iz = act.apply_and_transform(f_res_z);
        int iw = act.apply_and_transform(f_res_w);
        *(reinterpret_cast<int4*>(&dst[idx])) = ::make_int4(ix, iy, iz, iw);
    }
#endif
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
