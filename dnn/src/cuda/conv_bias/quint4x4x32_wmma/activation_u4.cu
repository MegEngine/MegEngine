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
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma/activation_u4.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <algorithm>
#include "./activation_u4.cuh"

namespace megdnn {
namespace cuda {
using namespace activation_u4;

namespace {

__host__ __device__ __forceinline__ int4 operator+(int4 lval, int4 rval) {
    return make_int4(lval.x + rval.x, lval.y + rval.y, lval.z + rval.z,
                     lval.w + rval.w);
}

template <typename ActivationOp>
__global__ void kern_activation_u4(int32_t* dst, const int32_t* zp_data,
                                   const int32_t* zp_filter,
                                   int32_t zp_data_filter, int batch_size,
                                   int OC, int OH, int OW,
                                   BiasVisitor visitor) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z * blockDim.z + threadIdx.z;
    constexpr int subbytes_per_pixel = 8;
    constexpr int load_width = 4;
    const int oc_blks = OC / subbytes_per_pixel;
    const int batch = bc / oc_blks;
    const int oc_blk = bc % oc_blks;

    int32_t* dptr = dst + batch * OC * OH * OW +
                    oc_blk * OH * OW * subbytes_per_pixel +
                    oh * OW * subbytes_per_pixel + ow * subbytes_per_pixel;
    if (batch >= batch_size || oh >= OH || ow >= OW)
        return;
    int32_t zp_data_val = zp_data[batch * OH * OW + oh * OW + ow];
    int32_t scalar = zp_data_val + zp_data_filter;
    int4 scalar4 = make_int4(scalar, scalar, scalar, scalar);
#pragma unroll
    for (int i = 0; i < subbytes_per_pixel / load_width; i++) {
        // do 128 bit load
        int4 zp_filter_val = *reinterpret_cast<const int4*>(
                zp_filter + oc_blk * subbytes_per_pixel + i * load_width);
        int4 bias_val = *reinterpret_cast<const int4*>(
                visitor.ptr(batch, oc_blk, oh, ow, i * load_width));
        int4 dst_val = *(reinterpret_cast<int4*>(dptr));
        int4 ret = dst_val + zp_filter_val + bias_val + scalar4;
        *(reinterpret_cast<int4*>(dptr)) = ActivationOp::apply(ret);
        dptr += load_width;
    }
}

}  // namespace

template <typename ActivationOp>
void _do_dispatch_activation_u4(int32_t* dst, BiasVisitor visitor,
                                const int32_t* zp_data,
                                const int32_t* zp_filter,
                                int32_t zp_data_filter, int batch_size, int co,
                                int ho, int wo, cudaStream_t stream) {
    void (*fptr)(int32_t*, const int32_t*, const int32_t*, int32_t, int, int OC,
                 int, int, BiasVisitor) = kern_activation_u4<ActivationOp>;
    dim3 grids{0, 0, 0};
    dim3 blocks{0, 0, 0};
    get_launch_config(reinterpret_cast<const void*>(fptr), wo, ho,
                      batch_size * co / 8, blocks, grids);
    kern_activation_u4<ActivationOp><<<grids, blocks, 0, stream>>>(
            dst, zp_data, zp_filter, zp_data_filter, batch_size, co, ho, wo,
            visitor);
    after_kernel_launch();
}

#define INST(_op)                                                             \
    template void _do_dispatch_activation_u4<_op>(                            \
            int32_t * dst, BiasVisitor visitor, const int32_t* zp_data,       \
            const int32_t* zp_filter, int32_t zp_data_filter, int batch_size, \
            int co, int ho, int wo, cudaStream_t stream);

INST(ActivationRELU);
INST(ActivationIdentity);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen
