/**
 * \file dnn/src/cuda/remap/forward.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include <cuda.h>
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

template <typename ctype>
struct DirectSrcVisitor {
    const ctype* ptr;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        return ptr + batch * im_size;
    }

    void move_batch(size_t batch, size_t im_size) { ptr += batch * im_size; }
};

template <const uint32_t format>
__device__ inline int get_offset(int height, int width, int channel, int h,
                                 int w, int c);

template <>
__device__ inline int get_offset<param_enumv::Remap::Format::NCHW>(
        int height, int width, int channel, int h, int w, int c) {
    return channel * h * w + height * w + width;
}

template <>
__device__ inline int get_offset<param_enumv::Remap::Format::NHWC>(
        int height, int width, int channel, int h, int w, int c) {
    return height * w * c + width * c + channel;
}

template <typename ctype, const uint32_t format, ::BorderMode bmode>
struct GetSrcData {
    __device__ static inline ctype get(const ctype* src, int height, int width,
                                       int channel, int h, int w, int c,
                                       float) {
        height = megcv::border_interpolate<bmode>(height, h);
        width = megcv::border_interpolate<bmode>(width, w);
        return src[get_offset<format>(height, width, channel, h, w, c)];
    }
};

template <typename ctype, const uint32_t format>
struct GetSrcData<ctype, format, ::BorderMode::BORDER_CONSTANT> {
    __device__ static inline ctype get(const ctype* src, int height, int width,
                                       int channel, int h, int w, int c,
                                       float scalar) {
        RoundingConverter<ctype> round_converter;
        return (height >= 0 && height < h && width >= 0 && width < w)
                       ? src[get_offset<format>(height, width, channel, h, w,
                                                c)]
                       : round_converter(scalar);
    }
};

template <typename ctype, typename SrcVisitor, ::BorderMode bmode>
__global__ void kern_general(SrcVisitor src, const float* map_xy,
                             ctype* __restrict dst, int C, int IH, int IW,
                             int OH, int OW, int S_IN, int S_IC, int S_IH,
                             int S_IW, float scalar) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, S_IN);
    dst += blockIdx.z * C * OH * OW;
    map_xy += blockIdx.z * 2 * OH * OW;
    RoundingConverter<ctype> round_converter;

    if (ow < OW && oh < OH) {
        float index_col = map_xy[oh * OW * 2 + ow * 2 + 0];
        float index_row = map_xy[oh * OW * 2 + ow * 2 + 1];
        int col = (int)floor(index_col);
        int row = (int)floor(index_row);
        float v = index_col - col;
        float u = index_row - row;
        for (int c = 0; c < C; ++c) {
            ctype a00 = GetSrcData<ctype, param_enumv::Remap::Format::NCHW,
                                   bmode>::get(sptr, row + 0, col + 0, c, IH,
                                               IW, C, scalar);
            ctype a01 = GetSrcData<ctype, param_enumv::Remap::Format::NCHW,
                                   bmode>::get(sptr, row + 0, col + 1, c, IH,
                                               IW, C, scalar);
            ctype a10 = GetSrcData<ctype, param_enumv::Remap::Format::NCHW,
                                   bmode>::get(sptr, row + 1, col + 0, c, IH,
                                               IW, C, scalar);
            ctype a11 = GetSrcData<ctype, param_enumv::Remap::Format::NCHW,
                                   bmode>::get(sptr, row + 1, col + 1, c, IH,
                                               IW, C, scalar);
            dst[get_offset<param_enumv::Remap::Format::NCHW>(oh, ow, c, OH, OW,
                                                             C)] =
                    round_converter(a00 * (1.f - u) * (1.f - v) +
                                    a01 * (1.f - u) * v + a10 * (1.f - v) * u +
                                    a11 * u * v);
        }
    }
}

template <typename ctype, typename SrcVisitor, ::BorderMode bmode>
__global__ void kern_general_nhwc(SrcVisitor src, const float* map_xy,
                                  ctype* __restrict dst, int C, int IH, int IW,
                                  int OH, int OW, float scalar) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;
    map_xy += blockIdx.z * 2 * OH * OW;
    RoundingConverter<ctype> round_converter;

    if (ow < OW && oh < OH) {
        float index_col = map_xy[oh * OW * 2 + ow * 2 + 0];
        float index_row = map_xy[oh * OW * 2 + ow * 2 + 1];
        int col = (int)floor(index_col);
        int row = (int)floor(index_row);
        float v = index_col - col;
        float u = index_row - row;
        for (int c = 0; c < C; ++c) {
            ctype a00 = GetSrcData<ctype, param_enumv::Remap::Format::NHWC,
                                   bmode>::get(sptr, row + 0, col + 0, c, IH,
                                               IW, C, scalar);
            ctype a01 = GetSrcData<ctype, param_enumv::Remap::Format::NHWC,
                                   bmode>::get(sptr, row + 0, col + 1, c, IH,
                                               IW, C, scalar);
            ctype a10 = GetSrcData<ctype, param_enumv::Remap::Format::NHWC,
                                   bmode>::get(sptr, row + 1, col + 0, c, IH,
                                               IW, C, scalar);
            ctype a11 = GetSrcData<ctype, param_enumv::Remap::Format::NHWC,
                                   bmode>::get(sptr, row + 1, col + 1, c, IH,
                                               IW, C, scalar);
            dst[get_offset<param_enumv::Remap::Format::NHWC>(oh, ow, c, OH, OW,
                                                             C)] =
                    round_converter(a00 * (1.f - u) * (1.f - v) +
                                    a01 * (1.f - u) * v + a10 * (1.f - v) * u +
                                    a11 * u * v);
        }
    }
}

template <typename ctype, typename SrcVisitor, const uint32_t format,
          ::BorderMode bmode>
void dispatch_with_visitor(SrcVisitor src, const float* map_xy, ctype* dst,
                           int N, int C, int IH, int IW, int OH, int OW,
                           float scalar, int S_IN, int S_IC, int S_IH, int S_IW,
                           cudaStream_t stream) {
    const int BX = 32, BY = 16;

    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        if (format == param_enumv::Remap::Format::NCHW) {
            kern_general<ctype, SrcVisitor, bmode>
                    <<<blocks, threads, 0, stream>>>(src, map_xy, dst, C, IH,
                                                     IW, OH, OW, S_IN, S_IC,
                                                     S_IH, S_IW, scalar);
        } else if (format == param_enumv::Remap::Format::NHWC) {
            kern_general_nhwc<ctype, SrcVisitor, bmode>
                    <<<blocks, threads, 0, stream>>>(src, map_xy, dst, C, IH,
                                                     IW, OH, OW, scalar);
        }

        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        dst += curr_batch_size * C * OH * OW;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace remap {

template <typename ctype, const uint32_t format, ::BorderMode bmode>
void forward_proxy(const ctype* src, const float* map_xy, ctype* dst, int N,
                   int C, int IH, int IW, int OH, int OW, float scalar,
                   int S_IN, int S_IC, int S_IH, int S_IW,
                   cudaStream_t stream) {
    DirectSrcVisitor<ctype> visitor;
    visitor.ptr = src;
    using SrcVisitor = DirectSrcVisitor<ctype>;
    dispatch_with_visitor<ctype, SrcVisitor, format, bmode>(
            visitor, map_xy, dst, N, C, IH, IW, OH, OW, scalar, S_IN, S_IC,
            S_IH, S_IW, stream);
    after_kernel_launch();
}

#define INST(ctype, format, bmode)                                           \
    template void forward_proxy<ctype, param_enumv::Remap::Format::format,   \
                                ::BorderMode::bmode>(                        \
            const ctype* src, const float*, ctype*, int, int, int, int, int, \
            int, float, int, int, int, int, cudaStream_t);

#define FOR_FORMAT_BMODE(ctype)           \
    INST(ctype, NCHW, BORDER_CONSTANT)    \
    INST(ctype, NCHW, BORDER_REPLICATE)   \
    INST(ctype, NCHW, BORDER_REFLECT)     \
    INST(ctype, NCHW, BORDER_REFLECT_101) \
    INST(ctype, NCHW, BORDER_WRAP)        \
    INST(ctype, NHWC, BORDER_CONSTANT)    \
    INST(ctype, NHWC, BORDER_REPLICATE)   \
    INST(ctype, NHWC, BORDER_REFLECT)     \
    INST(ctype, NHWC, BORDER_REFLECT_101) \
    INST(ctype, NHWC, BORDER_WRAP)

FOR_FORMAT_BMODE(float)
MEGDNN_INC_FLOAT16(FOR_FORMAT_BMODE(dt_float16))
FOR_FORMAT_BMODE(int8_t)
FOR_FORMAT_BMODE(uint8_t)

#undef FOR_BMODE
#undef INST
}  // namespace remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
