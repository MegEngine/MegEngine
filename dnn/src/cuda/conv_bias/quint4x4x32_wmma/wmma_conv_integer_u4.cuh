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
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma/wmma_conv_integer_u4.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include <cuda.h>
#if CUDA_VERSION >= 10000
#include <mma.h>
#endif

namespace megdnn {
namespace cuda {
namespace wmma_conv_integer_subbyte {

constexpr size_t WARP_SIZE = 32;
constexpr size_t WMMA_M = 8;
constexpr size_t WMMA_N = 8;
constexpr size_t WMMA_K = 32;
constexpr size_t IC_BLK = WMMA_K / 8;
constexpr size_t SKEW = 32;

template <int FH_ = 3, int FW_ = 3, int SH_ = 1, int SW_ = 1>
struct ConvConfig {
    static int const FH = FH_;
    static int const FW = FW_;
    static int const SH = SH_;
    static int const SW = SW_;
};

void _do_wmma_conv_integer_subbyte_1xfw(const uint8_t* d_data,
                                        const uint8_t* d_filter, int32_t* d_out,
                                        uint8_t* workspace, int batch_size,
                                        int hi, int wi, int ho, int wo, int ph,
                                        int pw, int ci, int co, int fh, int fw,
                                        int sh, int sw, uint8_t zp_data,
                                        cudaStream_t stream);

void _do_wmma_conv_integer_subbyte_fhxfw(const uint8_t* d_data,
                                         const uint8_t* d_filter,
                                         int32_t* d_out, int batch_size, int hi,
                                         int wi, int ho, int wo, int ph, int pw,
                                         int ci, int co, int fh, int fw, int sh,
                                         int sw, uint8_t zp_data,
                                         cudaStream_t stream);

}  // namespace wmma_conv_integer_subbyte
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
