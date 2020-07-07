/**
 * \file dnn/src/cuda/remap/backward_mat.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include <cuda_runtime.h>
#include "src/common/rounding_converter.cuh"
#include "src/cuda/cv/kernel_common.cuh"
#include "src/cuda/remap/common.h"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace remap;
using namespace rounding;

namespace {

template <const uint32_t format>
__device__ inline int get_offset(int height, int width, int channel, int h,
                                 int w, int c);

template <>
__device__ inline int get_offset<param_enumv::Remap::Format::NCHW>(
        int height, int width, int channel, int h, int w, int c) {
    return channel * h * w + height * w + width;
}

template <typename ctype, const uint32_t format, ::BorderMode bmode>
struct GetSrcData {
    __device__ static inline int get_index(int height, int width, int channel,
                                           int h, int w, int c) {
        height = megcv::border_interpolate<bmode>(height, h);
        width = megcv::border_interpolate<bmode>(width, w);
        return get_offset<format>(height, width, channel, h, w, c);
    }
};

template <typename ctype, const uint32_t format>
struct GetSrcData<ctype, format, ::BorderMode::BORDER_CONSTANT> {
    __device__ static inline int get_index(int height, int width, int channel,
                                           int h, int w, int c) {
        return (height >= 0 && height < h && width >= 0 && width < w)
                       ? get_offset<format>(height, width, channel, h, w, c)
                       : -1;
    }
};

template <typename ctype, const uint32_t format, ::BorderMode bmode>
__global__ void kern_general(const ctype* src, const float* map_xy,
                             const ctype* diff, float* __restrict grad, int C,
                             int IH, int IW, int OH, int OW, float scalar) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    src += blockIdx.z * C * IH * IW;
    diff += blockIdx.z * C * OH * OW;
    map_xy += blockIdx.z * 2 * OH * OW;
    grad += blockIdx.z * 2 * OH * OW;
    RoundingConverter<ctype> round_converter;

    if (ow < OW && oh < OH) {
        float index_col = map_xy[oh * OW * 2 + ow * 2 + 0];
        float index_row = map_xy[oh * OW * 2 + ow * 2 + 1];
        int col = static_cast<int>(floor(index_col));
        int row = static_cast<int>(floor(index_row));
        float v = index_col - col;  // alphaw
        float u = index_row - row;  // alphah
        const float one = 1.f;
        for (int c = 0; c < C; ++c) {
            float hidden = static_cast<float>(
                    diff[get_offset<format>(
                            oh, ow, c, OH, OW, C)]);
            float du = 0.f, dv = 0.f;

            int a00 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 0, col + 0, c, IH, IW, C);
            int a01 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 0, col + 1, c, IH, IW, C);
            int a10 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 1, col + 0, c, IH, IW, C);
            int a11 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 1, col + 1, c, IH, IW, C);

            dv -= ((a00 != -1) ? src[a00] : scalar) * (one - u);
            dv += ((a01 != -1) ? src[a01] : scalar) * (one - u);
            dv -= ((a10 != -1) ? src[a10] : scalar) * u;
            dv += ((a11 != -1) ? src[a11] : scalar) * u;

            du -= ((a00 != -1) ? src[a00] : scalar) * (one - v);
            du -= ((a01 != -1) ? src[a01] : scalar) * v;
            du += ((a10 != -1) ? src[a10] : scalar) * (one - v);
            du += ((a11 != -1) ? src[a11] : scalar) * v;

            grad[oh * OW * 2 + ow * 2 + 0] += round_converter(hidden * dv);
            grad[oh * OW * 2 + ow * 2 + 1] += round_converter(hidden * du);
        }
    }
}

template <typename ctype, const uint32_t format, ::BorderMode bmode>
void dispatch_backwardmat(const ctype* src, const float* map_xy,
                          const ctype* diff, float* grad, int N, int C, int IH,
                          int IW, int OH, int OW, float scalar,
                          cudaStream_t stream) {
    const int BX = 32, BY = 16;
    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        cuda_check(cudaMemsetAsync(
                grad, 0, sizeof(float) * curr_batch_size * OH * OW * 2,
                stream));
        kern_general<ctype, format, bmode><<<blocks, threads, 0, stream>>>(
                src, map_xy, diff, grad, C, IH, IW, OH, OW, scalar);

        N -= curr_batch_size;
        src += curr_batch_size * C * IH * IW;
        diff += curr_batch_size * C * OH * OW;
        map_xy += curr_batch_size * 2 * OH * OW;
        grad += curr_batch_size * 2 * OH * OW;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace remap {

template <typename ctype, const uint32_t format, ::BorderMode bmode>
void backwardmat_proxy(const ctype* src, const float* map_xy, const ctype* diff,
                       float* grad, int N, int C, int IH, int IW, int OH,
                       int OW, float scalar, cudaStream_t stream) {
    dispatch_backwardmat<ctype, format, bmode>(src, map_xy, diff, grad, N, C,
                                               IH, IW, OH, OW, scalar, stream);
    after_kernel_launch();
}

#define INST(ctype, format, bmode)                                             \
    template void backwardmat_proxy<ctype, param_enumv::Remap::Format::format, \
                                    ::BorderMode::bmode>(                      \
            const ctype*, const float*, const ctype*, float*, int, int, int,   \
            int, int, int, float, cudaStream_t);

#define FOR_FORMAT_BMODE(ctype)           \
    INST(ctype, NCHW, BORDER_CONSTANT)    \
    INST(ctype, NCHW, BORDER_REPLICATE)   \
    INST(ctype, NCHW, BORDER_REFLECT)     \
    INST(ctype, NCHW, BORDER_REFLECT_101) \
    INST(ctype, NCHW, BORDER_WRAP)

FOR_FORMAT_BMODE(float)
MEGDNN_INC_FLOAT16(FOR_FORMAT_BMODE(dt_bfloat16))

#undef FOR_FORMAT_BMODE
#undef INST

}  // namespace remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
