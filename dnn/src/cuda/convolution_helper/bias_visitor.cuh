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
 * \file dnn/src/cuda/convolution_helper/bias_visitor.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {

struct PerChannelBiasVisitor {
    const int32_t* __restrict__ bias;
#if MEGDNN_CC_CUDA
    __host__ __device__ __forceinline__ void move(int, int ch, int, int) { bias += ch; }
    __host__ __device__ __forceinline__ float4 at(int, int ch, int, int) {
        int ix = *(bias + ch);
        int iy = *(bias + ch + 1);
        int iz = *(bias + ch + 2);
        int iw = *(bias + ch + 3);
        return ::make_float4(
                static_cast<float>(ix), static_cast<float>(iy), static_cast<float>(iz),
                static_cast<float>(iw));
    }
    __host__ __device__ __forceinline__ float4 at(int, int ch, int) {
        int ix = *(bias + ch);
        int iy = *(bias + ch + 1);
        int iz = *(bias + ch + 2);
        int iw = *(bias + ch + 3);
        return ::make_float4(
                static_cast<float>(ix), static_cast<float>(iy), static_cast<float>(iz),
                static_cast<float>(iw));
    }
#endif
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
