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
 * \file dnn/src/cuda/convolution_helper/layout.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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

enum Format { CHWN4, CHWN16, NCHW4 };

template <Format format>
struct Layout;

template <>
struct Layout<Format::CHWN4> {
    static constexpr Format format = Format::CHWN4;
    int batch_stride;
    int channel_stride;
    int height_stride;
    int width_stride;

    __host__ __device__ __forceinline__ void init(const int batch,
                                                  const int /* channel */,
                                                  const int height,
                                                  const int width) {
        batch_stride = 4;
        channel_stride = height * width * batch * 4;
        height_stride = width * batch * 4;
        width_stride = batch * 4;
    }

    __device__ __forceinline__ size_t offset(const int batch, const int channel,
                                             const int height,
                                             const int width) {
        return batch * batch_stride + (channel >> 2) * channel_stride +
               height * height_stride + width * width_stride;
    }
};

template <>
struct Layout<Format::CHWN16> {
    static constexpr Format format = Format::CHWN16;
    int batch_stride;
    int channel_stride;
    int height_stride;
    int width_stride;

    __host__ __device__ __forceinline__ void init(const int batch,
                                                  const int /* channel */,
                                                  const int height,
                                                  const int width) {
        batch_stride = 16;
        channel_stride = height * width * batch * 16;
        height_stride = width * batch * 16;
        width_stride = batch * 16;
    }

    __device__ __forceinline__ size_t offset(const int batch, const int channel,
                                             const int height,
                                             const int width) {
        return batch * batch_stride + (channel >> 4) * channel_stride +
               height * height_stride + width * width_stride;
    }
};

template <>
struct Layout<Format::NCHW4> {
    static constexpr Format format = Format::NCHW4;
    int batch_stride;
    int channel_stride;
    int height_stride;
    int width_stride;

    __host__ __device__ __forceinline__ void init(const int /* batch */,
                                                  const int channel,
                                                  const int height,
                                                  const int width) {
        batch_stride = channel * height * width;
        channel_stride = height * width * 4;
        height_stride = width * 4;
        width_stride = 4;
    }

    __device__ __forceinline__ size_t offset(const int batch, const int channel,
                                             const int height,
                                             const int width) {
        return batch * batch_stride + (channel >> 2) * channel_stride +
               height * height_stride + width * width_stride;
    }
};

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
