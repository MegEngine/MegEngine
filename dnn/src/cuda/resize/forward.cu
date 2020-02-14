/**
 * \file dnn/src/cuda/resize/forward.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/resize/common.cuh"
#include "src/cuda/resize/common.h"
#include "src/common/rounding_converter.cuh"

#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace resize;

namespace {

template <typename ctype>
struct DirectSrcVisitor {
    const ctype* ptr;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        return ptr + batch * im_size;
    }

    void move_batch(size_t batch, size_t im_size) { ptr += batch * im_size; }
};

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general(SrcVisitor src, ctype* __restrict dst, int C,
                             int IH, int IW, int OH, int OW, int S_IN, int S_IC,
                             int S_IH, int S_IW, float scale_h, float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, S_IN);
    dst += blockIdx.z * C * OH * OW;

    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        for (int c = 0; c < C; ++c) {
            dst[oh * OW + ow] = output_converter(
                    sptr[ih0 * S_IH + iw0 * S_IW] * (1.0f - alphaw) *
                            (1.0f - alphah) +
                    sptr[ih0 * S_IH + iw1 * S_IW] * alphaw * (1.0f - alphah) +
                    sptr[ih1 * S_IH + iw0 * S_IW] * (1.0f - alphaw) * alphah +
                    sptr[ih1 * S_IH + iw1 * S_IW] * alphaw * alphah);

            sptr += S_IC;
            dst += OH * OW;
        }
    }
}

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general_nhwc(SrcVisitor src, ctype* __restrict dst, int C,
                                  int IH, int IW, int OH, int OW, float scale_h,
                                  float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;
    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        for (int c = 0; c < C; ++c) {
            dst[(oh * OW + ow) * C + c] = output_converter(
                    sptr[(ih0 * IW + iw0) * C + c] * (1.0f - alphaw) *
                            (1.0f - alphah) +
                    sptr[(ih0 * IW + iw1) * C + c] * alphaw * (1.0f - alphah) +
                    sptr[(ih1 * IW + iw0) * C + c] * (1.0f - alphaw) * alphah +
                    sptr[(ih1 * IW + iw1) * C + c] * alphaw * alphah);
        }
    }
}

template <typename ctype, typename SrcVisitor>
void dispatch_with_visitor(bool is_nhwc, SrcVisitor src, ctype* dst, int N,
                           int C, int IH, int IW, int OH, int OW, int S_IN,
                           int S_IC, int S_IH, int S_IW, cudaStream_t stream) {
    const int BY = 16, BX = 32;

    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        float scale_h = static_cast<float>(OH) / IH;
        float scale_w = static_cast<float>(OW) / IW;
        if (is_nhwc) {
            kern_general_nhwc<ctype, SrcVisitor,
                              rounding::RoundingConverter<ctype>>
                    <<<blocks, threads, 0, stream>>>(src, dst, C, IH, IW, OH,
                                                     OW, scale_h, scale_w);
        } else {
            kern_general<ctype, SrcVisitor, rounding::RoundingConverter<ctype>>
                    <<<blocks, threads, 0, stream>>>(src, dst, C, IH, IW, OH,
                                                     OW, S_IN, S_IC, S_IH, S_IW,
                                                     scale_h, scale_w);
        }
        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        dst += curr_batch_size * C * OH * OW;
    }
}

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general_nchw4(SrcVisitor src, ctype* __restrict dst, int C,
                             int IH, int IW, int OH, int OW, float scale_h,
                             float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;

    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        int o_coor = (oh * OW + ow) << 2;
        int i_coor00 = (ih0 * IW + iw0) << 2;
        int i_coor01 = (ih0 * IW + iw1) << 2;
        int i_coor10 = (ih1 * IW + iw0) << 2;
        int i_coor11 = (ih1 * IW + iw1) << 2;
        for (int c0 = 0, nr_chan = C >> 2; c0 < nr_chan; ++c0) {
#pragma unroll
            for (int c1 = 0; c1 < 4; ++c1) {
                dst[o_coor + c1] = output_converter(
                    sptr[i_coor00 + c1] * (1.0f - alphaw) * (1.0f - alphah) +
                    sptr[i_coor01 + c1] * alphaw * (1.0f - alphah) +
                    sptr[i_coor10 + c1] * (1.0f - alphaw) * alphah +
                    sptr[i_coor11 + c1] * alphaw * alphah);
            }
            dst += OH * OW * 4;
            sptr += IH * IW * 4;
        }
    }
}

template <typename ctype, typename SrcVisitor>
void dispatch_with_visitor_nchw4(SrcVisitor src, ctype* dst, int N, int C,
                                 int IH, int IW, int OH, int OW,
                                 cudaStream_t stream) {
    const int BY = 16, BX = 32;

    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        float scale_h = static_cast<float>(OH) / IH;
        float scale_w = static_cast<float>(OW) / IW;
        kern_general_nchw4<ctype, SrcVisitor,
                           rounding::RoundingConverter<ctype>>
                <<<blocks, threads, 0, stream>>>(src, dst, C, IH, IW, OH, OW,
                                                 scale_h, scale_w);
        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        dst += curr_batch_size * C * OH * OW;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace resize {

template <typename ctype>
void forward_proxy(bool is_nhwc, const ctype* src, ctype* dst, int N, int C,
                   int IH, int IW, int OH, int OW, int S_IN, int S_IC, int S_IH,
                   int S_IW, cudaStream_t stream) {
    DirectSrcVisitor<ctype> visitor;
    visitor.ptr = src;
    dispatch_with_visitor(is_nhwc, visitor, dst, N, C, IH, IW, OH, OW, S_IN,
                          S_IC, S_IH, S_IW, stream);
    after_kernel_launch();
}

template <typename ctype>
void forward_proxy_nchw4(const ctype* src, ctype* dst, int N, int C, int IH,
                         int IW, int OH, int OW, cudaStream_t stream) {
    DirectSrcVisitor<ctype> visitor;
    visitor.ptr = src;
    dispatch_with_visitor_nchw4(visitor, dst, N, C, IH, IW, OH, OW, stream);
    after_kernel_launch();
}

#define INST(ctype)                                                        \
    template void forward_proxy(bool, const ctype*, ctype*, int, int, int, \
                                int, int, int, int, int, int, int,         \
                                cudaStream_t);
INST(float)
INST(uint8_t)
INST(int8_t)
#undef INST

#define INST(ctype) \
    template void forward_proxy_nchw4(const ctype*, ctype*, int, int, int, \
                                int, int, int, cudaStream_t)

INST(int8_t);
#undef INST
}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
