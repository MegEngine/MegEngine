/**
 * \file dnn/src/cuda/warp_perspective/forward.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/warp_perspective/common.h"

#include "src/cuda/utils.cuh"
#include "src/cuda/warp_perspective/common.cuh"
#include "src/cuda/error_info.cuh"
#include "src/common/rounding_converter.cuh"
#include "megdnn/dtype.h"
#include <cstdio>

using namespace megdnn;
using namespace cuda;
using namespace warp_perspective;

namespace {

template<typename ctype>
struct DirectSrcVisitor {
    const ctype* ptr;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        return ptr + static_cast<int64_t>(batch) * static_cast<int64_t>(im_size);
    }

    void move_batch(size_t batch, size_t im_size) {
        ptr += batch * im_size;
    }
};

template<typename ctype>
struct IndexedSrcVisitor {
    const ctype* ptr;
    const int* idx;
    int N_SRC;

    AsyncErrorInfo* error_info;
    void* error_tracker;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        int orig_batch = batch;
        batch = idx[batch];
        if (batch < 0 || batch >= N_SRC) {
            set_async_error_info(error_info, error_tracker,
                    "mat_idx out of bound: mat_idx[%d]=%d src_batch=%d",
                    orig_batch, batch, N_SRC);
            batch = 0;
        }
        return ptr + static_cast<int64_t>(batch) * static_cast<int64_t>(im_size);
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
    mat += blockIdx.z * 3*3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6]*ow + mat[7]*oh + mat[8];
        float iw = (mat[0]*ow + mat[1]*oh + mat[2]) / denominator;
        float ih = (mat[3]*ow + mat[4]*oh + mat[5]) / denominator;
        int iw0 = getter(floor(iw) + 0, IW);
        int iw1 = getter(floor(iw) + 1, IW);
        int ih0 = getter(floor(ih) + 0, IH);
        int ih1 = getter(floor(ih) + 1, IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        for (int c = 0; c < C; ++c) {
            dst[oh * OW + ow] =
                    output_converter(sptr[ih0 * IW + iw0] * nalpha * nbeta +
                                     sptr[ih0 * IW + iw1] * nalpha * pbeta +
                                     sptr[ih1 * IW + iw0] * palpha * nbeta +
                                     sptr[ih1 * IW + iw1] * palpha * pbeta);
            sptr += IH*IW;
            dst += OH*OW;
        }
    }
}

template <typename ctype, typename Getter, typename SrcVisitor,
          typename OutputConverter>
__global__ void kern_general_nchw4(SrcVisitor src, const float* __restrict mat,
                             ctype* __restrict dst, int C, int IH, int IW,
                             int OH, int OW) {
    Getter getter;
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;
    mat += blockIdx.z * 3 * 3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6] * ow + mat[7] * oh + mat[8];
        float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;
        float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;
        int iw0 = getter(floor(iw) + 0, IW);
        int iw1 = getter(floor(iw) + 1, IW);
        int ih0 = getter(floor(ih) + 0, IH);
        int ih1 = getter(floor(ih) + 1, IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        int o_coor = (oh * OW + ow) << 2;
        int i_coor_00 = (ih0 * IW + iw0) << 2;
        int i_coor_01 = (ih0 * IW + iw1) << 2;
        int i_coor_10 = (ih1 * IW + iw0) << 2;
        int i_coor_11 = (ih1 * IW + iw1) << 2;
        for (int c0 = 0, nr_chan = C / 4; c0 < nr_chan; ++c0) {
#pragma unroll
            for (int c1 = 0; c1 < 4; ++c1) {
                dst[o_coor + c1] =
                        output_converter(sptr[i_coor_00 + c1] * nalpha * nbeta +
                                         sptr[i_coor_01 + c1] * nalpha * pbeta +
                                         sptr[i_coor_10 + c1] * palpha * nbeta +
                                         sptr[i_coor_11 + c1] * palpha * pbeta);
            }
            sptr += IH * IW * 4;
            dst += OH * OW * 4;
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
    mat += blockIdx.z * 3*3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6]*ow + mat[7]*oh + mat[8];
        float iw = (mat[0]*ow + mat[1]*oh + mat[2]) / denominator;
        float ih = (mat[3]*ow + mat[4]*oh + mat[5]) / denominator;
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

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_const_border_nchw4(SrcVisitor src,
                                        const float* __restrict mat,
                                        ctype* __restrict dst, int C, int IH,
                                        int IW, int OH, int OW, ctype bval) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;
    mat += blockIdx.z * 3 * 3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6] * ow + mat[7] * oh + mat[8];
        float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;
        float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;
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
        int i_coor_00 = (ih0 * IW + iw0) << 2;
        int i_coor_01 = (ih0 * IW + iw1) << 2;
        int i_coor_10 = (ih1 * IW + iw0) << 2;
        int i_coor_11 = (ih1 * IW + iw1) << 2;
        int o_coor = (oh * OW + ow) << 2;
        for (int c0 = 0, nr_chan = C / 4; c0 < nr_chan; ++c0) {
#pragma unroll
            for (int c1 = 0; c1 < 4; ++c1) {
                ctype v00 = (okh0 && okw0 ? sptr[i_coor_00 + c1] : bval);
                ctype v01 = (okh0 && okw1 ? sptr[i_coor_01 + c1] : bval);
                ctype v10 = (okh1 && okw0 ? sptr[i_coor_10 + c1] : bval);
                ctype v11 = (okh1 && okw1 ? sptr[i_coor_11 + c1] : bval);
                ctype val = output_converter(
                        v00 * nalpha * nbeta + v01 * nalpha * pbeta +
                        v10 * palpha * nbeta + v11 * palpha * pbeta);
                dst[o_coor + c1] = val;
            }
            sptr += IH * IW * 4;
            dst += OH * OW * 4;
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
    mat += blockIdx.z * 3 * 3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6] * ow + mat[7] * oh + mat[8];
        float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;
        float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;
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
    mat += blockIdx.z * 3*3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6]*ow + mat[7]*oh + mat[8];
        float iw = (mat[0]*ow + mat[1]*oh + mat[2]) / denominator;
        float ih = (mat[3]*ow + mat[4]*oh + mat[5]) / denominator;
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
#undef DISPATCH
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
}

template <typename ctype, typename SrcVisitor>
void dispatch_with_visitor_nchw4(SrcVisitor src, const float* mat, ctype* dst,
                                 int N, int C, int IH, int IW, int OH, int OW,
                                 ctype bval, BorderMode bmode,
                                 cudaStream_t stream) {
    const int BY = 16, BX = 32;
#define DISPATCH(Getter)                                                       \
    do {                                                                       \
        kern_general_nchw4<ctype, Getter, SrcVisitor,                          \
                           rounding::RoundingConverter<ctype>>                 \
                <<<blocks, threads, 0, stream>>>(src, mat, dst, C, IH, IW, OH, \
                                                 OW);                          \
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
#undef DISPATCH
            case BORDER_CONSTANT:
                kern_const_border_nchw4<ctype, SrcVisitor,
                                        rounding::RoundingConverter<ctype>>
                        <<<blocks, threads, 0, stream>>>(src, mat, dst, C, IH,
                                                         IW, OH, OW, bval);
                break;
            default:
                break;
        }

        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        mat += curr_batch_size * 3 * 3;
        dst += curr_batch_size * C * OH * OW;
    }
}

} // anonymous namespace

namespace megdnn {
namespace cuda {
namespace warp_perspective {

template<typename ctype>
void forward_proxy(
        bool is_nhwc,
        const ctype *src, const float *mat, const int *mat_idx,
        ctype *dst, int N_SRC, int N_MAT,
        int C, int IH, int IW, int OH, int OW, ctype bval,
        BorderMode bmode,
        megcore::AsyncErrorInfo* error_info, void* error_tracker,
        cudaStream_t stream)
{
    if (mat_idx) {
        IndexedSrcVisitor<ctype> visitor;
        visitor.ptr = src;
        visitor.idx = mat_idx;
        visitor.N_SRC = N_SRC;
        visitor.error_info = error_info;
        visitor.error_tracker = error_tracker;
        dispatch_with_visitor(is_nhwc,
                visitor, mat, dst, N_MAT, C, IH, IW, OH, OW, bval,
                bmode, stream);
    } else {
        DirectSrcVisitor<ctype> visitor;
        visitor.ptr = src;
        dispatch_with_visitor(is_nhwc,
                visitor, mat, dst, N_MAT, C, IH, IW, OH, OW, bval,
                bmode, stream);
    }
    after_kernel_launch();
}

template <typename ctype>
void forward_proxy_nchw4(const ctype* src, const float* mat, const int* mat_idx,
                         ctype* dst, int N_SRC, int N_MAT, int C, int IH,
                         int IW, int OH, int OW, ctype bval, BorderMode bmode,
                         megcore::AsyncErrorInfo* error_info,
                         void* error_tracker, cudaStream_t stream) {
    if (mat_idx) {
        IndexedSrcVisitor<ctype> visitor;
        visitor.ptr = src;
        visitor.idx = mat_idx;
        visitor.N_SRC = N_SRC;
        visitor.error_info = error_info;
        visitor.error_tracker = error_tracker;
        dispatch_with_visitor_nchw4(visitor, mat, dst, N_MAT, C, IH, IW, OH, OW,
                                    bval, bmode, stream);
    } else {
        DirectSrcVisitor<ctype> visitor;
        visitor.ptr = src;
        dispatch_with_visitor_nchw4(visitor, mat, dst, N_MAT, C, IH, IW, OH, OW,
                                    bval, bmode, stream);
    }
    after_kernel_launch();
}

#define INST(ctype)                                                           \
    template void forward_proxy(bool, const ctype*, const float*, const int*, \
                                ctype*, int, int, int, int, int, int, int,    \
                                ctype, BorderMode, megcore::AsyncErrorInfo*,  \
                                void*, cudaStream_t);
INST(float)
INST(uint8_t)
#ifndef MEGDNN_DISABLE_FLOAT16
INST(dt_float16)
#endif
INST(int8_t)
#undef INST

#define INST(ctype)                                                          \
    template void forward_proxy_nchw4(                                       \
            const ctype*, const float*, const int*, ctype*, int, int, int,   \
            int, int, int, int, ctype, BorderMode, megcore::AsyncErrorInfo*, \
            void*, cudaStream_t);

INST(int8_t)
#undef INST

} // namespace warp_perspective
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
