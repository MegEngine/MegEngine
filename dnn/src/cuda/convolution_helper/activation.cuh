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
 * \file dnn/src/cuda/convolution_helper/activation.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {
template <uint32_t mode>
struct Activation;

#define DEF_APPLY_AND_TRANSFORM(_act)                               \
    __device__ __forceinline__ int apply_and_transform(float4 in) { \
        return transform_float4_to_int8x4(                          \
                quantize(_act::apply(dequantize(in))));             \
    }

template <>
struct Activation<megdnn::param_enumv::ConvBias::NonlineMode::H_SWISH> {
    float scale;
    float inv_scale;
    MEGDNN_HOST MEGDNN_DEVICE Activation(float scale, float inv_scale)
            : scale{scale}, inv_scale{inv_scale} {}
#if MEGDNN_CC_CUDA
    DEF_APPLY_AND_TRANSFORM(
            Activation<megdnn::param_enumv::ConvBias::NonlineMode::H_SWISH>);
    __device__ __forceinline__ float4 dequantize(float4 in) {
        return scale * in;
    }
    __device__ __forceinline__ float4 quantize(float4 in) {
        return inv_scale * in;
    }
    __device__ __forceinline__ static float4 apply(float4 in) {
        float x = in.x * fminf(fmaxf(in.x + 3.f, 0.f), 6.f) * (1.f / 6.f);
        float y = in.y * fminf(fmaxf(in.y + 3.f, 0.f), 6.f) * (1.f / 6.f);
        float z = in.z * fminf(fmaxf(in.z + 3.f, 0.f), 6.f) * (1.f / 6.f);
        float w = in.w * fminf(fmaxf(in.w + 3.f, 0.f), 6.f) * (1.f / 6.f);
        return make_float4(x, y, z, w);
    }
#endif
};

template <>
struct Activation<megdnn::param_enumv::ConvBias::NonlineMode::RELU> {
    MEGDNN_HOST MEGDNN_DEVICE Activation(float /* scale */,
                                         float /* inv_scale */) {}
#if MEGDNN_CC_CUDA
    DEF_APPLY_AND_TRANSFORM(
            Activation<megdnn::param_enumv::ConvBias::NonlineMode::RELU>);
    __device__ __forceinline__ float4 dequantize(float4 in) { return in; }
    __device__ __forceinline__ float4 quantize(float4 in) { return in; }
    __device__ __forceinline__ static float4 apply(float4 in) {
        float x = in.x <= 0 ? 0 : in.x;
        float y = in.y <= 0 ? 0 : in.y;
        float z = in.z <= 0 ? 0 : in.z;
        float w = in.w <= 0 ? 0 : in.w;
        return make_float4(x, y, z, w);
    }
#endif
};

template <>
struct Activation<megdnn::param_enumv::ConvBias::NonlineMode::IDENTITY> {
    MEGDNN_HOST MEGDNN_DEVICE Activation(float /* scale */,
                                         float /* inv_scale */) {}
#if MEGDNN_CC_CUDA
    DEF_APPLY_AND_TRANSFORM(
            Activation<megdnn::param_enumv::ConvBias::NonlineMode::IDENTITY>);
    __device__ __forceinline__ float4 dequantize(float4 in) { return in; }
    __device__ __forceinline__ float4 quantize(float4 in) { return in; }
    __device__ __forceinline__ static float4 apply(float4 in) { return in; }
#endif
};
#undef DEF_APPLY_AND_TRANSFORM

#define MEGDNN_FOREACH_NONLINE_MODE(cb) cb(H_SWISH) cb(RELU) cb(IDENTITY)

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
