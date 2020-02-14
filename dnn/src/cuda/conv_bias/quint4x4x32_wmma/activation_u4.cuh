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
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma/activation_u4.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace activation_u4 {

void get_launch_config(const void* kern, int dimx, int dimy, int dimz,
                       dim3& blocks, dim3& grids);

struct BiasVisitor {
    const int32_t* bias_ptr;
    int batch_stride;
    int channel_stride;
    int height_stride;
    int width_stride;
#ifdef MEGDNN_CC_CUDA
    __host__ __device__ __forceinline__ const int32_t* ptr(int batch,
                                                           int oc_blk, int oh,
                                                           int ow,
                                                           int oc_remain) {
        return bias_ptr + batch * batch_stride + oc_blk * channel_stride +
               oh * height_stride + ow * width_stride + oc_remain;
    }
#endif
};

struct ActivationRELU {
#ifdef MEGDNN_CC_CUDA
    __host__ __device__ __forceinline__ static int4 apply(int4 in) {
        int4 ret;
        ret.x = in.x <= 0 ? 0 : in.x;
        ret.y = in.y <= 0 ? 0 : in.y;
        ret.z = in.z <= 0 ? 0 : in.z;
        ret.w = in.w <= 0 ? 0 : in.w;
        return ret;
    }
#endif
};

struct ActivationIdentity {
#ifdef MEGDNN_CC_CUDA
    __host__ __device__ __forceinline__ static int4 apply(int4 in) {
        return in;
    }
#endif
};
}  // namespace activation_u4

template <typename ActivationOp>
void _do_dispatch_activation_u4(int32_t* dst,
                                activation_u4::BiasVisitor visitor,
                                const int32_t* zp_data,
                                const int32_t* zp_filter,
                                int32_t zp_data_filter, int batch_size, int co,
                                int ho, int wo, cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen
