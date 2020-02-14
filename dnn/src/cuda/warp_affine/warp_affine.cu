/**
 * \file dnn/src/cuda/warp_affine/warp_affine.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/warp_affine/common.h"

#include "src/cuda/utils.cuh"
#include "src/cuda/warp_affine/common.cuh"
#include "src/common/rounding_converter.cuh"
#include <cstdio>

using namespace megdnn;
using namespace cuda;
using namespace warp_affine;

namespace {

template<typename ctype>
struct DirectSrcVisitor {
    const ctype* ptr;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        return ptr + batch * im_size;
    }

    void move_batch(size_t batch, size_t im_size) {
        ptr += batch * im_size;
    }
};

template<typename ctype>
struct IndexedSrcVisitor {
    const ctype* ptr;
    const int* idx;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        batch = idx[batch];
        return ptr + batch * im_size;
    }

    void move_batch(size_t batch, size_t) {
        idx += batch;
    }
};

template <typename ctype, typename Getter, typename SrcVisitor,
          typename OutputConverter>
__global__ void kern_general(SrcVisitor src, const float* __restrict mat,
                             ctype* __restrict dst, int C, int IH, int IW,
                             int OH, int OW) {
    Getter getter;
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C*OH*OW;
    mat += blockIdx.z * 2*3;
    if (ow < OW && oh < OH) {
        float iw = mat[0] * ow + mat[1] * oh + mat[2];
        float ih = mat[3] * ow + mat[4] * oh + mat[5];
        int iw0 = getter(floor(iw) + 0, IW);
        int iw1 = getter(floor(iw) + 1, IW);
        int ih0 = getter(floor(ih) + 0, IH);
        int ih1 = getter(floor(ih) + 1, IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        for (int c = 0; c < C; ++c) {
            dst[oh*OW+ow] = output_converter(
                sptr[ih0*IW+iw0]*nalpha*nbeta + sptr[ih0*IW+iw1]*nalpha*pbeta +
                sptr[ih1*IW+iw0]*palpha*nbeta + sptr[ih1*IW+iw1]*palpha*pbeta);
            sptr += IH*IW;
            dst += OH*OW;
        }
    }
}

template<typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_const_border(
        SrcVisitor src, const float *__restrict mat, ctype *__restrict dst,
        int C, int IH, int IW, int OH, int OW, ctype bval)
{
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C*OH*OW;
    mat += blockIdx.z * 2*3;
    if (ow < OW && oh < OH) {
        float iw = mat[0] * ow + mat[1] * oh + mat[2];
        float ih = mat[3] * ow + mat[4] * oh + mat[5];
        int iw0 = floor(iw) + 0;
        int iw1 = floor(iw) + 1;
        int ih0 = floor(ih) + 0;
        int ih1 = floor(ih) + 1;
        bool okw0 = (iw0 >= 0 && iw0 < IW);
        bool okw1 = (iw1 >= 0 && iw1 < IW);
        bool okh0 = (ih0 >= 0 && ih0 < IH);
        bool okh1 = (ih1 >= 0 && ih1 < IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        for (int c = 0; c < C; ++c) {
            ctype v00 = (okh0 && okw0 ? sptr[ih0*IW+iw0] : bval);
            ctype v01 = (okh0 && okw1 ? sptr[ih0*IW+iw1] : bval);
            ctype v10 = (okh1 && okw0 ? sptr[ih1*IW+iw0] : bval);
            ctype v11 = (okh1 && okw1 ? sptr[ih1*IW+iw1] : bval);
            ctype val = output_converter(
                v00*nalpha*nbeta + v01*nalpha*pbeta +
                v10*palpha*nbeta + v11*palpha*pbeta);
            dst[oh*OW+ow] = val;
            sptr += IH*IW;
            dst += OH*OW;
        }
    }
}

template <typename ctype, typename Getter, typename SrcVisitor,
          typename OutputConverter>
__global__ void kern_general_nhwc(SrcVisitor src, const float* __restrict mat,
                                  ctype* __restrict dst, int C, int IH, int IW,
                                  int OH, int OW) {
    Getter getter;
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;
    mat += blockIdx.z * 2 * 3;
    if (ow < OW && oh < OH) {
        float iw = mat[0] * ow + mat[1] * oh + mat[2];
        float ih = mat[3] * ow + mat[4] * oh + mat[5];
        int iw0 = getter(floor(iw) + 0, IW);
        int iw1 = getter(floor(iw) + 1, IW);
        int ih0 = getter(floor(ih) + 0, IH);
        int ih1 = getter(floor(ih) + 1, IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        for (int c = 0; c < C; ++c) {
            dst[(oh * OW + ow) * C + c] = output_converter(
                    sptr[(ih0 * IW + iw0) * C + c] * nalpha * nbeta +
                    sptr[(ih0 * IW + iw1) * C + c] * nalpha * pbeta +
                    sptr[(ih1 * IW + iw0) * C + c] * palpha * nbeta +
                    sptr[(ih1 * IW + iw1) * C + c] * palpha * pbeta);
        }
    }
}

template<typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_const_border_nhwc(
        SrcVisitor src, const float *__restrict mat, ctype *__restrict dst,
        int C, int IH, int IW, int OH, int OW, ctype bval)
{
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C*OH*OW;
    mat += blockIdx.z * 2*3;
    if (ow < OW && oh < OH) {
        float iw = mat[0] * ow + mat[1] * oh + mat[2];
        float ih = mat[3] * ow + mat[4] * oh + mat[5];
        int iw0 = floor(iw) + 0;
        int iw1 = floor(iw) + 1;
        int ih0 = floor(ih) + 0;
        int ih1 = floor(ih) + 1;
        bool okw0 = (iw0 >= 0 && iw0 < IW);
        bool okw1 = (iw1 >= 0 && iw1 < IW);
        bool okh0 = (ih0 >= 0 && ih0 < IH);
        bool okh1 = (ih1 >= 0 && ih1 < IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        for (int c = 0; c < C; ++c) {
            ctype v00 = (okh0 && okw0 ? sptr[(ih0*IW+iw0)*C+c] : bval);
            ctype v01 = (okh0 && okw1 ? sptr[(ih0*IW+iw1)*C+c] : bval);
            ctype v10 = (okh1 && okw0 ? sptr[(ih1*IW+iw0)*C+c] : bval);
            ctype v11 = (okh1 && okw1 ? sptr[(ih1*IW+iw1)*C+c] : bval);
            ctype val = output_converter(
                v00*nalpha*nbeta + v01*nalpha*pbeta +
                v10*palpha*nbeta + v11*palpha*pbeta);
            dst[(oh*OW+ow)*C+c] = val;
        }
    }
}

template <typename ctype, typename SrcVisitor>
void dispatch_with_visitor(bool is_nhwc, SrcVisitor src, const float* mat,
                           ctype* dst, int N, int C, int IH, int IW, int OH,
                           int OW, ctype bval, BorderMode bmode,
                           cudaStream_t stream) {
    const int BY = 16, BX = 32;
#define DISPATCH(Getter)                                                       \
    do {                                                                       \
        if (is_nhwc) {                                                         \
            kern_general_nhwc<ctype, Getter, SrcVisitor,                       \
                              rounding::RoundingConverter<ctype>>              \
                    <<<blocks, threads, 0, stream>>>(src, mat, dst, C, IH, IW, \
                                                     OH, OW);                  \
        } else {                                                               \
            kern_general<ctype, Getter, SrcVisitor,                            \
                         rounding::RoundingConverter<ctype>>                   \
                    <<<blocks, threads, 0, stream>>>(src, mat, dst, C, IH, IW, \
                                                     OH, OW);                  \
        }                                                                      \
    } while (0)

    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        switch (bmode) {
            case BORDER_REPLICATE:
                DISPATCH(ReplicateGetter);
                break;
            case BORDER_REFLECT:
                DISPATCH(ReflectGetter);
                break;
            case BORDER_REFLECT_101:
                DISPATCH(Reflect101Getter);
                break;
            case BORDER_WRAP:
                DISPATCH(WrapGetter);
                break;
            case BORDER_CONSTANT:
                if (is_nhwc) {
                    kern_const_border_nhwc<ctype, SrcVisitor,
                                           rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    src, mat, dst, C, IH, IW, OH, OW, bval);
                } else {
                    kern_const_border<ctype, SrcVisitor,
                                      rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    src, mat, dst, C, IH, IW, OH, OW, bval);
                }
                break;
            default:
                break;
        }

        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        mat += curr_batch_size * 3 * 3;
        dst += curr_batch_size * C * OH * OW;
    }

#undef DISPATCH
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace warp_affine {

template <typename ctype>
void forward_proxy(bool is_nhwc, const ctype* src, const float* mat, ctype* dst,
                   int N, int C, int IH, int IW, int OH, int OW, ctype bval,
                   BorderMode bmode, cudaStream_t stream) {
    DirectSrcVisitor<ctype> visitor;
    visitor.ptr = src;
    dispatch_with_visitor(is_nhwc, visitor, mat, dst, N, C, IH, IW, OH, OW,
                          bval, bmode, stream);
    after_kernel_launch();
}

#define INST(ctype)                                                            \
    template void forward_proxy(bool, const ctype*, const float*, ctype*, int, \
                                int, int, int, int, int, ctype, BorderMode,    \
                                cudaStream_t);
INST(float)
INST(uint8_t)
INST(int8_t)
#undef INST

}  // namespace warp_affine
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
