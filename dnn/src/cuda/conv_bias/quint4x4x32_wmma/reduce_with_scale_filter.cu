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
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma/reduce_with_scale_filter.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./reduce_with_scale_filter.cuh"
#include "src/cuda/reduce_helper.cuh"

using namespace megdnn;
using namespace cuda;

namespace {

struct ReduceWithScaleUInt4Op {
    typedef int32_t wtype;
    const uint8_t* src;
    int32_t* dst;
    int32_t scale;
    static const wtype INIT = 0;

#if MEGDNN_CC_CUDA
    __host__ __device__ void write(uint32_t idx, wtype val) {
        dst[idx] = val * scale;
    }

    __host__ __device__ static wtype apply(wtype a, wtype b) { return a + b; }

    __device__ wtype read(uint32_t idx) {
        constexpr uint32_t subbytes_per_pixel = 8;
        const uint32_t* sptr =
                (const uint32_t*)(src + subbytes_per_pixel * idx / 2);
        uint32_t val = *sptr;
        int32_t ret = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t cur = (val & 0xF);
            ret += cur;
            val = (val >> 4);
        }
        return ret;
    }
#endif
};

}  // namespace

void megdnn::cuda::_do_dispatch_reduce_with_scale_filter_u4(
        const uint8_t* src, int32_t scale, uint32_t rows, uint32_t cols,
        int32_t* dst, cudaStream_t stream) {
    // rows = OC
    // cols is measured in pixels, i.e. IC * FH * FW / 8, a pixel consists of 8
    // subbyte data,
    ReduceWithScaleUInt4Op op;
    op.src = src;
    op.scale = scale;
    op.dst = dst;
    static_cast<void>(op);
    static_cast<void>(stream);
    static_cast<void>(rows);
    static_cast<void>(cols);
    run_reduce<ReduceWithScaleUInt4Op, false>(dst + rows, rows, cols, 1, stream,
                                              op);
}

size_t megdnn::cuda::_do_dispatch_reduce_workspace_in_bytes(size_t A, size_t B,
                                                            size_t C) {
    return get_reduce_workspace_in_bytes<ReduceWithScaleUInt4Op>(A, B, C);
}

// vim: ft=cpp syntax=cuda.doxygen
