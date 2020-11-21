/**
 * \file dnn/src/cuda/warp_perspective/forward.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
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

template <typename ctype>
struct DirectSrcVisitor {
    const ctype* ptr;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        return ptr +
               static_cast<int64_t>(batch) * static_cast<int64_t>(im_size);
    }

    void move_batch(size_t batch, size_t im_size) { ptr += batch * im_size; }
};

template <typename ctype>
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
            set_async_error_info(
                    error_info, error_tracker,
                    "mat_idx out of bound: mat_idx[%d]=%d src_batch=%d",
                    orig_batch, batch, N_SRC);
            batch = 0;
        }
        return ptr +
               static_cast<int64_t>(batch) * static_cast<int64_t>(im_size);
    }

    void move_batch(size_t batch, size_t) { idx += batch; }
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
            dst[oh * OW + ow] =
                    output_converter(sptr[ih0 * IW + iw0] * nalpha * nbeta +
                                     sptr[ih0 * IW + iw1] * nalpha * pbeta +
                                     sptr[ih1 * IW + iw0] * palpha * nbeta +
                                     sptr[ih1 * IW + iw1] * palpha * pbeta);
            sptr += IH * IW;
            dst += OH * OW;
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

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_const_border(SrcVisitor src, const float* __restrict mat,
                                  ctype* __restrict dst, int C, int IH, int IW,
                                  int OH, int OW, ctype bval) {
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
        for (int c = 0; c < C; ++c) {
            ctype v00 = (okh0 && okw0 ? sptr[ih0 * IW + iw0] : bval);
            ctype v01 = (okh0 && okw1 ? sptr[ih0 * IW + iw1] : bval);
            ctype v10 = (okh1 && okw0 ? sptr[ih1 * IW + iw0] : bval);
            ctype v11 = (okh1 && okw1 ? sptr[ih1 * IW + iw1] : bval);
            ctype val = output_converter(
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +
                    v10 * palpha * nbeta + v11 * palpha * pbeta);
            dst[oh * OW + ow] = val;
            sptr += IH * IW;
            dst += OH * OW;
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

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_const_border_nhwc(SrcVisitor src,
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
        for (int c = 0; c < C; ++c) {
            ctype v00 = (okh0 && okw0 ? sptr[(ih0 * IW + iw0) * C + c] : bval);
            ctype v01 = (okh0 && okw1 ? sptr[(ih0 * IW + iw1) * C + c] : bval);
            ctype v10 = (okh1 && okw0 ? sptr[(ih1 * IW + iw0) * C + c] : bval);
            ctype v11 = (okh1 && okw1 ? sptr[(ih1 * IW + iw1) * C + c] : bval);
            ctype val = output_converter(
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +
                    v10 * palpha * nbeta + v11 * palpha * pbeta);
            dst[(oh * OW + ow) * C + c] = val;
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

template <typename SrcType, typename DstType>
struct CudaTypeCvt;

template <>
struct CudaTypeCvt<dt_quint8, int8_t> {
    CudaDTypeParamImpl<dt_quint8> m_src_param;
    CudaTypeCvt(CudaDTypeParamImpl<dt_quint8> src_param) {
        m_src_param = src_param;
    };
    inline __device__ int8_t operator()(uint8_t val) {
        return val - m_src_param.zero_point;
    }
};

template <>
struct CudaTypeCvt<dt_quint8, float> {
    CudaDTypeParamImpl<dt_quint8> m_src_param;
    CudaTypeCvt(CudaDTypeParamImpl<dt_quint8> src_param) {
        m_src_param = src_param;
    };
    __device__ __forceinline__ float operator()(uint8_t val) {
        return m_src_param.dequantize(dt_quint8(val));
    }
};

#define INST(dst_ctype, vec_dst_type)                                          \
    template <typename src_dtype, typename src_ctype, typename Getter,         \
              typename SrcVisitor>                                             \
    __global__ void kern_general_quint8_nhw_nchw4(                             \
            SrcVisitor src, const float* __restrict mat,                       \
            dst_ctype* __restrict dst, int IH, int IW, int OH, int OW,         \
            CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {                      \
        Getter getter;                                                         \
        rounding::RoundingConverter<src_ctype> warp_out_converter;             \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                        \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                        \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, IH * IW);       \
        dst += blockIdx.z * OH * OW * 4;                                       \
        mat += blockIdx.z * 3 * 3;                                             \
        if (ow < OW && oh < OH) {                                              \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];            \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;     \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;     \
            int iw0 = getter(floor(iw) + 0, IW);                               \
            int iw1 = getter(floor(iw) + 1, IW);                               \
            int ih0 = getter(floor(ih) + 0, IH);                               \
            int ih1 = getter(floor(ih) + 1, IH);                               \
            float palpha = ih - floor(ih);                                     \
            float pbeta = iw - floor(iw);                                      \
            float nalpha = 1.0f - palpha;                                      \
            float nbeta = 1.0f - pbeta;                                        \
            vec_dst_type result;                                               \
            src_ctype val_x =                                                  \
                    warp_out_converter(sptr[ih0 * IW + iw0] * nalpha * nbeta + \
                                       sptr[ih0 * IW + iw1] * nalpha * pbeta + \
                                       sptr[ih1 * IW + iw0] * palpha * nbeta + \
                                       sptr[ih1 * IW + iw1] * palpha * pbeta); \
            result.x = type_cvt(val_x);                                        \
            result.y = result.z = result.w = 0;                                \
            *((vec_dst_type*)dst + oh * OW + ow) = result;                     \
        }                                                                      \
    }

INST(int8_t, char4)
#undef INST

#define INST(dst_ctype, vec_dst_type)                                      \
    template <typename src_dtype, typename src_ctype, typename SrcVisitor> \
    __global__ void kern_const_border_quint8_nhw_nchw4(                    \
            SrcVisitor src, const float* __restrict mat,                   \
            dst_ctype* __restrict dst, int IH, int IW, int OH, int OW,     \
            src_ctype bval, CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {  \
        rounding::RoundingConverter<src_ctype> warp_out_converter;         \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                    \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                    \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, IH * IW);   \
        dst += blockIdx.z * OH * OW * 4;                                   \
        mat += blockIdx.z * 3 * 3;                                         \
        if (ow < OW && oh < OH) {                                          \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];        \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator; \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator; \
            int iw0 = floor(iw) + 0;                                       \
            int iw1 = floor(iw) + 1;                                       \
            int ih0 = floor(ih) + 0;                                       \
            int ih1 = floor(ih) + 1;                                       \
            bool okw0 = (iw0 >= 0 && iw0 < IW);                            \
            bool okw1 = (iw1 >= 0 && iw1 < IW);                            \
            bool okh0 = (ih0 >= 0 && ih0 < IH);                            \
            bool okh1 = (ih1 >= 0 && ih1 < IH);                            \
            float palpha = ih - floor(ih);                                 \
            float pbeta = iw - floor(iw);                                  \
            float nalpha = 1.0f - palpha;                                  \
            float nbeta = 1.0f - pbeta;                                    \
            vec_dst_type result;                                           \
            src_ctype v00 = (okh0 && okw0 ? sptr[ih0 * IW + iw0] : bval);  \
            src_ctype v01 = (okh0 && okw1 ? sptr[ih0 * IW + iw1] : bval);  \
            src_ctype v10 = (okh1 && okw0 ? sptr[ih1 * IW + iw0] : bval);  \
            src_ctype v11 = (okh1 && okw1 ? sptr[ih1 * IW + iw1] : bval);  \
            src_ctype val_x = warp_out_converter(                          \
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +          \
                    v10 * palpha * nbeta + v11 * palpha * pbeta);          \
            result.x = type_cvt(val_x);                                    \
            result.y = result.z = result.w = 0;                            \
            *((vec_dst_type*)dst + oh * OW + ow) = result;                 \
        }                                                                  \
    }

INST(int8_t, char4)
#undef INST

#define INST(dst_ctype, vec_dst_type)                                          \
    template <typename src_dtype, typename src_ctype, typename Getter,         \
              typename SrcVisitor>                                             \
    __global__ void kern_general_quint8_n3hw_nchw4(                            \
            SrcVisitor src, const float* __restrict mat,                       \
            dst_ctype* __restrict dst, int IH, int IW, int OH, int OW,         \
            CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {                      \
        Getter getter;                                                         \
        rounding::RoundingConverter<src_ctype> warp_out_converter;             \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                        \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                        \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, 3 * IH * IW);   \
        dst += blockIdx.z * OH * OW * 4;                                       \
        mat += blockIdx.z * 3 * 3;                                             \
        if (ow < OW && oh < OH) {                                              \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];            \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;     \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;     \
            int iw0 = getter(floor(iw) + 0, IW);                               \
            int iw1 = getter(floor(iw) + 1, IW);                               \
            int ih0 = getter(floor(ih) + 0, IH);                               \
            int ih1 = getter(floor(ih) + 1, IH);                               \
            float palpha = ih - floor(ih);                                     \
            float pbeta = iw - floor(iw);                                      \
            float nalpha = 1.0f - palpha;                                      \
            float nbeta = 1.0f - pbeta;                                        \
            vec_dst_type result;                                               \
            src_ctype val_x =                                                  \
                    warp_out_converter(sptr[ih0 * IW + iw0] * nalpha * nbeta + \
                                       sptr[ih0 * IW + iw1] * nalpha * pbeta + \
                                       sptr[ih1 * IW + iw0] * palpha * nbeta + \
                                       sptr[ih1 * IW + iw1] * palpha * pbeta); \
            src_ctype val_y = warp_out_converter(                              \
                    sptr[IW * IH + ih0 * IW + iw0] * nalpha * nbeta +          \
                    sptr[IW * IH + ih0 * IW + iw1] * nalpha * pbeta +          \
                    sptr[IW * IH + ih1 * IW + iw0] * palpha * nbeta +          \
                    sptr[IW * IH + ih1 * IW + iw1] * palpha * pbeta);          \
            src_ctype val_z = warp_out_converter(                              \
                    sptr[2 * IW * IH + ih0 * IW + iw0] * nalpha * nbeta +      \
                    sptr[2 * IW * IH + ih0 * IW + iw1] * nalpha * pbeta +      \
                    sptr[2 * IW * IH + ih1 * IW + iw0] * palpha * nbeta +      \
                    sptr[2 * IW * IH + ih1 * IW + iw1] * palpha * pbeta);      \
            result.x = type_cvt(val_x);                                        \
            result.y = type_cvt(val_y);                                        \
            result.z = type_cvt(val_z);                                        \
            result.w = 0;                                                      \
            *((vec_dst_type*)dst + oh * OW + ow) = result;                     \
        }                                                                      \
    }

INST(int8_t, char4)
#undef INST

#define INST(dst_ctype, vec_dst_type)                                         \
    template <typename src_dtype, typename src_ctype, typename SrcVisitor>    \
    __global__ void kern_const_border_quint8_n3hw_nchw4(                      \
            SrcVisitor src, const float* __restrict mat,                      \
            dst_ctype* __restrict dst, int IH, int IW, int OH, int OW,        \
            src_ctype bval, CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {     \
        rounding::RoundingConverter<src_ctype> warp_out_converter;            \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                       \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                       \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, 3 * IH * IW);  \
        dst += blockIdx.z * OH * OW * 4;                                      \
        mat += blockIdx.z * 3 * 3;                                            \
        if (ow < OW && oh < OH) {                                             \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];           \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;    \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;    \
            int iw0 = floor(iw) + 0;                                          \
            int iw1 = floor(iw) + 1;                                          \
            int ih0 = floor(ih) + 0;                                          \
            int ih1 = floor(ih) + 1;                                          \
            bool okw0 = (iw0 >= 0 && iw0 < IW);                               \
            bool okw1 = (iw1 >= 0 && iw1 < IW);                               \
            bool okh0 = (ih0 >= 0 && ih0 < IH);                               \
            bool okh1 = (ih1 >= 0 && ih1 < IH);                               \
            float palpha = ih - floor(ih);                                    \
            float pbeta = iw - floor(iw);                                     \
            float nalpha = 1.0f - palpha;                                     \
            float nbeta = 1.0f - pbeta;                                       \
            vec_dst_type result;                                              \
            src_ctype v00, v01, v10, v11;                                     \
            v00 = (okh0 && okw0 ? sptr[ih0 * IW + iw0] : bval);               \
            v01 = (okh0 && okw1 ? sptr[ih0 * IW + iw1] : bval);               \
            v10 = (okh1 && okw0 ? sptr[ih1 * IW + iw0] : bval);               \
            v11 = (okh1 && okw1 ? sptr[ih1 * IW + iw1] : bval);               \
            src_ctype val_x = warp_out_converter(                             \
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +             \
                    v10 * palpha * nbeta + v11 * palpha * pbeta);             \
            v00 = (okh0 && okw0 ? sptr[IH * IW + ih0 * IW + iw0] : bval);     \
            v01 = (okh0 && okw1 ? sptr[IH * IW + ih0 * IW + iw1] : bval);     \
            v10 = (okh1 && okw0 ? sptr[IH * IW + ih1 * IW + iw0] : bval);     \
            v11 = (okh1 && okw1 ? sptr[IH * IW + ih1 * IW + iw1] : bval);     \
            src_ctype val_y = warp_out_converter(                             \
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +             \
                    v10 * palpha * nbeta + v11 * palpha * pbeta);             \
            v00 = (okh0 && okw0 ? sptr[2 * IH * IW + ih0 * IW + iw0] : bval); \
            v01 = (okh0 && okw1 ? sptr[2 * IH * IW + ih0 * IW + iw1] : bval); \
            v10 = (okh1 && okw0 ? sptr[2 * IH * IW + ih1 * IW + iw0] : bval); \
            v11 = (okh1 && okw1 ? sptr[2 * IH * IW + ih1 * IW + iw1] : bval); \
            src_ctype val_z = warp_out_converter(                             \
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +             \
                    v10 * palpha * nbeta + v11 * palpha * pbeta);             \
            result.x = type_cvt(val_x);                                       \
            result.y = type_cvt(val_y);                                       \
            result.z = type_cvt(val_z);                                       \
            result.w = 0;                                                     \
            *((vec_dst_type*)dst + oh * OW + ow) = result;                    \
        }                                                                     \
    }

INST(int8_t, char4)
#undef INST

#define INST(dst_ctype, vec_dst_type)                                        \
    template <typename src_dtype, typename src_ctype, typename Getter,       \
              typename SrcVisitor>                                           \
    __global__ void kern_general_quint8_nhw3_nchw4(                          \
            SrcVisitor src, const float* __restrict mat,                     \
            dst_ctype* __restrict dst, int IH, int IW, int OH, int OW,       \
            CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {                    \
        Getter getter;                                                       \
        rounding::RoundingConverter<src_ctype> warp_out_converter;           \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                      \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                      \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, 3 * IH * IW); \
        dst += blockIdx.z * OH * OW * 4;                                     \
        mat += blockIdx.z * 3 * 3;                                           \
        if (ow < OW && oh < OH) {                                            \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];          \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;   \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;   \
            int iw0 = getter(floor(iw) + 0, IW);                             \
            int iw1 = getter(floor(iw) + 1, IW);                             \
            int ih0 = getter(floor(ih) + 0, IH);                             \
            int ih1 = getter(floor(ih) + 1, IH);                             \
            float palpha = ih - floor(ih);                                   \
            float pbeta = iw - floor(iw);                                    \
            float nalpha = 1.0f - palpha;                                    \
            float nbeta = 1.0f - pbeta;                                      \
            vec_dst_type result;                                             \
            src_ctype val_x = warp_out_converter(                            \
                    sptr[(ih0 * IW + iw0) * 3] * nalpha * nbeta +            \
                    sptr[(ih0 * IW + iw1) * 3] * nalpha * pbeta +            \
                    sptr[(ih1 * IW + iw0) * 3] * palpha * nbeta +            \
                    sptr[(ih1 * IW + iw1) * 3] * palpha * pbeta);            \
            src_ctype val_y = warp_out_converter(                            \
                    sptr[(ih0 * IW + iw0) * 3 + 1] * nalpha * nbeta +        \
                    sptr[(ih0 * IW + iw1) * 3 + 1] * nalpha * pbeta +        \
                    sptr[(ih1 * IW + iw0) * 3 + 1] * palpha * nbeta +        \
                    sptr[(ih1 * IW + iw1) * 3 + 1] * palpha * pbeta);        \
            src_ctype val_z = warp_out_converter(                            \
                    sptr[(ih0 * IW + iw0) * 3 + 2] * nalpha * nbeta +        \
                    sptr[(ih0 * IW + iw1) * 3 + 2] * nalpha * pbeta +        \
                    sptr[(ih1 * IW + iw0) * 3 + 2] * palpha * nbeta +        \
                    sptr[(ih1 * IW + iw1) * 3 + 2] * palpha * pbeta);        \
            result.x = type_cvt(val_x);                                      \
            result.y = type_cvt(val_y);                                      \
            result.z = type_cvt(val_z);                                      \
            result.w = 0;                                                    \
            *((vec_dst_type*)dst + oh * OW + ow) = result;                   \
        }                                                                    \
    }

INST(int8_t, char4)
#undef INST

#define INST(dst_ctype, vec_dst_type)                                        \
    template <typename src_dtype, typename src_ctype, typename SrcVisitor>   \
    __global__ void kern_const_border_quint8_nhw3_nchw4(                     \
            SrcVisitor src, const float* __restrict mat,                     \
            dst_ctype* __restrict dst, int IH, int IW, int OH, int OW,       \
            src_ctype bval, CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {    \
        rounding::RoundingConverter<src_ctype> warp_out_converter;           \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                      \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                      \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, 3 * IH * IW); \
        dst += blockIdx.z * OH * OW * 4;                                     \
        mat += blockIdx.z * 3 * 3;                                           \
        if (ow < OW && oh < OH) {                                            \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];          \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;   \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;   \
            int iw0 = floor(iw) + 0;                                         \
            int iw1 = floor(iw) + 1;                                         \
            int ih0 = floor(ih) + 0;                                         \
            int ih1 = floor(ih) + 1;                                         \
            bool okw0 = (iw0 >= 0 && iw0 < IW);                              \
            bool okw1 = (iw1 >= 0 && iw1 < IW);                              \
            bool okh0 = (ih0 >= 0 && ih0 < IH);                              \
            bool okh1 = (ih1 >= 0 && ih1 < IH);                              \
            float palpha = ih - floor(ih);                                   \
            float pbeta = iw - floor(iw);                                    \
            float nalpha = 1.0f - palpha;                                    \
            float nbeta = 1.0f - pbeta;                                      \
            vec_dst_type result;                                             \
            src_ctype v00, v01, v10, v11;                                    \
            v00 = (okh0 && okw0 ? sptr[(ih0 * IW + iw0) * 3] : bval);        \
            v01 = (okh0 && okw1 ? sptr[(ih0 * IW + iw1) * 3] : bval);        \
            v10 = (okh1 && okw0 ? sptr[(ih1 * IW + iw0) * 3] : bval);        \
            v11 = (okh1 && okw1 ? sptr[(ih1 * IW + iw1) * 3] : bval);        \
            src_ctype val_x = warp_out_converter(                            \
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +            \
                    v10 * palpha * nbeta + v11 * palpha * pbeta);            \
            v00 = (okh0 && okw0 ? sptr[(ih0 * IW + iw0) * 3 + 1] : bval);    \
            v01 = (okh0 && okw1 ? sptr[(ih0 * IW + iw1) * 3 + 1] : bval);    \
            v10 = (okh1 && okw0 ? sptr[(ih1 * IW + iw0) * 3 + 1] : bval);    \
            v11 = (okh1 && okw1 ? sptr[(ih1 * IW + iw1) * 3 + 1] : bval);    \
            src_ctype val_y = warp_out_converter(                            \
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +            \
                    v10 * palpha * nbeta + v11 * palpha * pbeta);            \
            v00 = (okh0 && okw0 ? sptr[(ih0 * IW + iw0) * 3 + 2] : bval);    \
            v01 = (okh0 && okw1 ? sptr[(ih0 * IW + iw1) * 3 + 2] : bval);    \
            v10 = (okh1 && okw0 ? sptr[(ih1 * IW + iw0) * 3 + 2] : bval);    \
            v11 = (okh1 && okw1 ? sptr[(ih1 * IW + iw1) * 3 + 2] : bval);    \
            src_ctype val_z = warp_out_converter(                            \
                    v00 * nalpha * nbeta + v01 * nalpha * pbeta +            \
                    v10 * palpha * nbeta + v11 * palpha * pbeta);            \
            result.x = type_cvt(val_x);                                      \
            result.y = type_cvt(val_y);                                      \
            result.z = type_cvt(val_z);                                      \
            result.w = 0;                                                    \
            *((vec_dst_type*)dst + oh * OW + ow) = result;                   \
        }                                                                    \
    }

INST(int8_t, char4)
#undef INST

template <typename src_dtype, typename src_ctype, typename dst_ctype,
          typename SrcVisitor>
void dispatch_with_visitor_quint8_dimshuffle_typecvt_nchw4(
        bool is_nhwc, SrcVisitor src, const float* mat, dst_ctype* dst, int N,
        int C, int IH, int IW, int OH, int OW, src_ctype bval,
        CudaDTypeParamImpl<src_dtype> param, BorderMode bmode,
        cudaStream_t stream) {
    const int BY = 16, BX = 32;
    CudaTypeCvt<src_dtype, dst_ctype> type_cvt(param);
#define DISPATCH(Getter)                                                    \
    do {                                                                    \
        if (C == 1) {                                                       \
            kern_general_quint8_nhw_nchw4<src_dtype, src_ctype, Getter,     \
                                          SrcVisitor>                       \
                    <<<blocks, threads, 0, stream>>>(src, mat, dst, IH, IW, \
                                                     OH, OW, type_cvt);     \
        } else if (is_nhwc) {                                               \
            kern_general_quint8_nhw3_nchw4<src_dtype, src_ctype, Getter,    \
                                           SrcVisitor>                      \
                    <<<blocks, threads, 0, stream>>>(src, mat, dst, IH, IW, \
                                                     OH, OW, type_cvt);     \
        } else {                                                            \
            kern_general_quint8_n3hw_nchw4<src_dtype, src_ctype, Getter,    \
                                           SrcVisitor>                      \
                    <<<blocks, threads, 0, stream>>>(src, mat, dst, IH, IW, \
                                                     OH, OW, type_cvt);     \
        }                                                                   \
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
                if (C == 1) {
                    kern_const_border_quint8_nhw_nchw4<src_dtype, src_ctype,
                                                       SrcVisitor>
                            <<<blocks, threads, 0, stream>>>(src, mat, dst, IH,
                                                             IW, OH, OW, bval,
                                                             type_cvt);
                } else if (is_nhwc) {
                    kern_const_border_quint8_nhw3_nchw4<src_dtype, src_ctype,
                                                        SrcVisitor>
                            <<<blocks, threads, 0, stream>>>(src, mat, dst, IH,
                                                             IW, OH, OW, bval,
                                                             type_cvt);
                } else {
                    kern_const_border_quint8_n3hw_nchw4<src_dtype, src_ctype,
                                                        SrcVisitor>
                            <<<blocks, threads, 0, stream>>>(src, mat, dst, IH,
                                                             IW, OH, OW, bval,
                                                             type_cvt);
                }
                break;
            default:
                break;
        }

        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        mat += curr_batch_size * 3 * 3;
        dst += curr_batch_size * 4 * OH * OW;
    }
}

#define INST(dst_ctype)                                                       \
    template <typename src_dtype, typename src_ctype, typename Getter,        \
              typename SrcVisitor>                                            \
    __global__ void kern_general_quint8_nchw(                                 \
            SrcVisitor src, const float* __restrict mat,                      \
            dst_ctype* __restrict dst, int C, int IH, int IW, int OH, int OW, \
            CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {                     \
        Getter getter;                                                        \
        rounding::RoundingConverter<src_ctype> warp_out_converter;            \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                       \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                       \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);  \
        dst += blockIdx.z * C * OH * OW;                                      \
        mat += blockIdx.z * 3 * 3;                                            \
        if (ow < OW && oh < OH) {                                             \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];           \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;    \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;    \
            int iw0 = getter(floor(iw) + 0, IW);                              \
            int iw1 = getter(floor(iw) + 1, IW);                              \
            int ih0 = getter(floor(ih) + 0, IH);                              \
            int ih1 = getter(floor(ih) + 1, IH);                              \
            float palpha = ih - floor(ih);                                    \
            float pbeta = iw - floor(iw);                                     \
            float nalpha = 1.0f - palpha;                                     \
            float nbeta = 1.0f - pbeta;                                       \
            for (int c = 0; c < C; ++c) {                                     \
                src_ctype val = warp_out_converter(                           \
                        sptr[ih0 * IW + iw0] * nalpha * nbeta +               \
                        sptr[ih0 * IW + iw1] * nalpha * pbeta +               \
                        sptr[ih1 * IW + iw0] * palpha * nbeta +               \
                        sptr[ih1 * IW + iw1] * palpha * pbeta);               \
                dst_ctype result;                                             \
                result = type_cvt(val);                                       \
                dst[oh * OW + ow] = result;                                   \
                sptr += IH * IW;                                              \
                dst += OH * OW;                                               \
            }                                                                 \
        }                                                                     \
    }

INST(float)
#undef INST

#define INST(dst_ctype)                                                       \
    template <typename src_dtype, typename src_ctype, typename SrcVisitor>    \
    __global__ void kern_const_border_quint8_nchw(                            \
            SrcVisitor src, const float* __restrict mat,                      \
            dst_ctype* __restrict dst, int C, int IH, int IW, int OH, int OW, \
            src_ctype bval, CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {     \
        rounding::RoundingConverter<src_ctype> warp_out_converter;            \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                       \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                       \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);  \
        dst += blockIdx.z * C * OH * OW;                                      \
        mat += blockIdx.z * 3 * 3;                                            \
        if (ow < OW && oh < OH) {                                             \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];           \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;    \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;    \
            int iw0 = floor(iw) + 0;                                          \
            int iw1 = floor(iw) + 1;                                          \
            int ih0 = floor(ih) + 0;                                          \
            int ih1 = floor(ih) + 1;                                          \
            bool okw0 = (iw0 >= 0 && iw0 < IW);                               \
            bool okw1 = (iw1 >= 0 && iw1 < IW);                               \
            bool okh0 = (ih0 >= 0 && ih0 < IH);                               \
            bool okh1 = (ih1 >= 0 && ih1 < IH);                               \
            float palpha = ih - floor(ih);                                    \
            float pbeta = iw - floor(iw);                                     \
            float nalpha = 1.0f - palpha;                                     \
            float nbeta = 1.0f - pbeta;                                       \
            for (int c = 0; c < C; ++c) {                                     \
                src_ctype v00 = (okh0 && okw0 ? sptr[ih0 * IW + iw0] : bval); \
                src_ctype v01 = (okh0 && okw1 ? sptr[ih0 * IW + iw1] : bval); \
                src_ctype v10 = (okh1 && okw0 ? sptr[ih1 * IW + iw0] : bval); \
                src_ctype v11 = (okh1 && okw1 ? sptr[ih1 * IW + iw1] : bval); \
                src_ctype val = warp_out_converter(                           \
                        v00 * nalpha * nbeta + v01 * nalpha * pbeta +         \
                        v10 * palpha * nbeta + v11 * palpha * pbeta);         \
                dst_ctype result;                                             \
                result = type_cvt(val);                                       \
                dst[oh * OW + ow] = result;                                   \
                sptr += IH * IW;                                              \
                dst += OH * OW;                                               \
            }                                                                 \
        }                                                                     \
    }

INST(float)
#undef INST

#define INST(dst_ctype)                                                       \
    template <typename src_dtype, typename src_ctype, typename Getter,        \
              typename SrcVisitor>                                            \
    __global__ void kern_general_quint8_nhwc_nchw(                            \
            SrcVisitor src, const float* __restrict mat,                      \
            dst_ctype* __restrict dst, int C, int IH, int IW, int OH, int OW, \
            CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {                     \
        Getter getter;                                                        \
        rounding::RoundingConverter<src_ctype> warp_out_converter;            \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                       \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                       \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);  \
        dst += blockIdx.z * C * OH * OW;                                      \
        mat += blockIdx.z * 3 * 3;                                            \
        if (ow < OW && oh < OH) {                                             \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];           \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;    \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;    \
            int iw0 = getter(floor(iw) + 0, IW);                              \
            int iw1 = getter(floor(iw) + 1, IW);                              \
            int ih0 = getter(floor(ih) + 0, IH);                              \
            int ih1 = getter(floor(ih) + 1, IH);                              \
            float palpha = ih - floor(ih);                                    \
            float pbeta = iw - floor(iw);                                     \
            float nalpha = 1.0f - palpha;                                     \
            float nbeta = 1.0f - pbeta;                                       \
            for (int c = 0; c < C; ++c) {                                     \
                src_ctype val = warp_out_converter(                           \
                        sptr[(ih0 * IW + iw0) * C + c] * nalpha * nbeta +     \
                        sptr[(ih0 * IW + iw1) * C + c] * nalpha * pbeta +     \
                        sptr[(ih1 * IW + iw0) * C + c] * palpha * nbeta +     \
                        sptr[(ih1 * IW + iw1) * C + c] * palpha * pbeta);     \
                dst_ctype result;                                             \
                result = type_cvt(val);                                       \
                dst[oh * OW + ow] = result;                                   \
                dst += OH * OW;                                               \
            }                                                                 \
        }                                                                     \
    }

INST(float)
#undef INST

#define INST(dst_ctype)                                                        \
    template <typename src_dtype, typename src_ctype, typename SrcVisitor>     \
    __global__ void kern_const_border_quint8_nhwc_nchw(                        \
            SrcVisitor src, const float* __restrict mat,                       \
            dst_ctype* __restrict dst, int C, int IH, int IW, int OH, int OW,  \
            src_ctype bval, CudaTypeCvt<src_dtype, dst_ctype> type_cvt) {      \
        rounding::RoundingConverter<src_ctype> warp_out_converter;             \
        int ow = blockIdx.x * blockDim.x + threadIdx.x;                        \
        int oh = blockIdx.y * blockDim.y + threadIdx.y;                        \
        const src_ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);   \
        dst += blockIdx.z * C * OH * OW;                                       \
        mat += blockIdx.z * 3 * 3;                                             \
        if (ow < OW && oh < OH) {                                              \
            float denominator = mat[6] * ow + mat[7] * oh + mat[8];            \
            float iw = (mat[0] * ow + mat[1] * oh + mat[2]) / denominator;     \
            float ih = (mat[3] * ow + mat[4] * oh + mat[5]) / denominator;     \
            int iw0 = floor(iw) + 0;                                           \
            int iw1 = floor(iw) + 1;                                           \
            int ih0 = floor(ih) + 0;                                           \
            int ih1 = floor(ih) + 1;                                           \
            bool okw0 = (iw0 >= 0 && iw0 < IW);                                \
            bool okw1 = (iw1 >= 0 && iw1 < IW);                                \
            bool okh0 = (ih0 >= 0 && ih0 < IH);                                \
            bool okh1 = (ih1 >= 0 && ih1 < IH);                                \
            float palpha = ih - floor(ih);                                     \
            float pbeta = iw - floor(iw);                                      \
            float nalpha = 1.0f - palpha;                                      \
            float nbeta = 1.0f - pbeta;                                        \
            for (int c = 0; c < C; ++c) {                                      \
                src_ctype v00 = (okh0 && okw0 ? sptr[(ih0 * IW + iw0) * C + c] \
                                              : bval);                         \
                src_ctype v01 = (okh0 && okw1 ? sptr[(ih0 * IW + iw1) * C + c] \
                                              : bval);                         \
                src_ctype v10 = (okh1 && okw0 ? sptr[(ih1 * IW + iw0) * C + c] \
                                              : bval);                         \
                src_ctype v11 = (okh1 && okw1 ? sptr[(ih1 * IW + iw1) * C + c] \
                                              : bval);                         \
                float val = warp_out_converter(                                \
                        v00 * nalpha * nbeta + v01 * nalpha * pbeta +          \
                        v10 * palpha * nbeta + v11 * palpha * pbeta);          \
                dst_ctype result;                                              \
                result = type_cvt(val);                                        \
                dst[oh * OW + ow] = result;                                    \
                dst += OH * OW;                                                \
            }                                                                  \
        }                                                                      \
    }

INST(float)
#undef INST

template <typename src_dtype, typename src_ctype, typename dst_ctype,
          typename SrcVisitor>
void dispatch_with_visitor_quint8_dimshuffle_typecvt_nchw(
        bool is_nhwc, SrcVisitor src, const float* mat, dst_ctype* dst, int N,
        int C, int IH, int IW, int OH, int OW, src_ctype bval,
        CudaDTypeParamImpl<src_dtype> param, BorderMode bmode,
        cudaStream_t stream) {
    const int BY = 16, BX = 32;
    CudaTypeCvt<src_dtype, dst_ctype> type_cvt(param);
#define DISPATCH(Getter)                                                       \
    do {                                                                       \
        if (is_nhwc) {                                                         \
            kern_general_quint8_nhwc_nchw<src_dtype, src_ctype, Getter,        \
                                          SrcVisitor>                          \
                    <<<blocks, threads, 0, stream>>>(src, mat, dst, C, IH, IW, \
                                                     OH, OW, type_cvt);        \
        } else {                                                               \
            kern_general_quint8_nchw<src_dtype, src_ctype, Getter, SrcVisitor> \
                    <<<blocks, threads, 0, stream>>>(src, mat, dst, C, IH, IW, \
                                                     OH, OW, type_cvt);        \
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
                    kern_const_border_quint8_nhwc_nchw<src_dtype, src_ctype,
                                                       SrcVisitor>
                            <<<blocks, threads, 0, stream>>>(src, mat, dst, C,
                                                             IH, IW, OH, OW,
                                                             bval, type_cvt);
                } else {
                    kern_const_border_quint8_nchw<src_dtype, src_ctype,
                                                  SrcVisitor>
                            <<<blocks, threads, 0, stream>>>(src, mat, dst, C,
                                                             IH, IW, OH, OW,
                                                             bval, type_cvt);
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

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace warp_perspective {

template <typename ctype>
void forward_proxy(bool is_nhwc, const ctype* src, const float* mat,
                   const int* mat_idx, ctype* dst, int N_SRC, int N_MAT, int C,
                   int IH, int IW, int OH, int OW, ctype bval, BorderMode bmode,
                   megcore::AsyncErrorInfo* error_info, void* error_tracker,
                   cudaStream_t stream) {
    if (mat_idx) {
        IndexedSrcVisitor<ctype> visitor;
        visitor.ptr = src;
        visitor.idx = mat_idx;
        visitor.N_SRC = N_SRC;
        visitor.error_info = error_info;
        visitor.error_tracker = error_tracker;
        dispatch_with_visitor(is_nhwc, visitor, mat, dst, N_MAT, C, IH, IW, OH,
                              OW, bval, bmode, stream);
    } else {
        DirectSrcVisitor<ctype> visitor;
        visitor.ptr = src;
        dispatch_with_visitor(is_nhwc, visitor, mat, dst, N_MAT, C, IH, IW, OH,
                              OW, bval, bmode, stream);
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

template <typename src_dtype, typename src_ctype, typename dst_ctype>
void forward_proxy_quint8_dimshuffle_typecvt_nchw4(
        bool is_nhwc, const src_ctype* src, const float* mat,
        const int* mat_idx, dst_ctype* dst, int N_SRC, int N_MAT, int C, int IH,
        int IW, int OH, int OW, src_ctype bval, DTypeParamImpl<src_dtype> param,
        BorderMode bmode, megcore::AsyncErrorInfo* error_info,
        void* error_tracker, cudaStream_t stream) {
    CudaDTypeParamImpl<src_dtype> dtype_param(param);
    if (mat_idx) {
        IndexedSrcVisitor<src_ctype> visitor;
        visitor.ptr = src;
        visitor.idx = mat_idx;
        visitor.N_SRC = N_SRC;
        visitor.error_info = error_info;
        visitor.error_tracker = error_tracker;
        dispatch_with_visitor_quint8_dimshuffle_typecvt_nchw4(
                is_nhwc, visitor, mat, dst, N_MAT, C, IH, IW, OH, OW, bval,
                dtype_param, bmode, stream);
    } else {
        DirectSrcVisitor<src_ctype> visitor;
        visitor.ptr = src;
        dispatch_with_visitor_quint8_dimshuffle_typecvt_nchw4(
                is_nhwc, visitor, mat, dst, N_MAT, C, IH, IW, OH, OW, bval,
                dtype_param, bmode, stream);
    }
    after_kernel_launch();
}

#define INST(src_dtype, src_ctype, dst_ctype)                         \
    template void forward_proxy_quint8_dimshuffle_typecvt_nchw4(      \
            bool is_nhwc, const src_ctype*, const float*, const int*, \
            dst_ctype*, int, int, int, int, int, int, int, src_ctype, \
            DTypeParamImpl<src_dtype> param, BorderMode,              \
            megcore::AsyncErrorInfo*, void*, cudaStream_t);

INST(dt_quint8, uint8_t, int8_t)
#undef INST

template <typename src_dtype, typename src_ctype, typename dst_ctype>
void forward_proxy_quint8_dimshuffle_typecvt_nchw(
        bool is_nhwc, const src_ctype* src, const float* mat,
        const int* mat_idx, dst_ctype* dst, int N_SRC, int N_MAT, int C, int IH,
        int IW, int OH, int OW, src_ctype bval, DTypeParamImpl<src_dtype> param,
        BorderMode bmode, megcore::AsyncErrorInfo* error_info,
        void* error_tracker, cudaStream_t stream) {
    CudaDTypeParamImpl<src_dtype> dtype_param(param);
    if (mat_idx) {
        IndexedSrcVisitor<src_ctype> visitor;
        visitor.ptr = src;
        visitor.idx = mat_idx;
        visitor.N_SRC = N_SRC;
        visitor.error_info = error_info;
        visitor.error_tracker = error_tracker;
        dispatch_with_visitor_quint8_dimshuffle_typecvt_nchw(
                is_nhwc, visitor, mat, dst, N_MAT, C, IH, IW, OH, OW, bval,
                dtype_param, bmode, stream);
    } else {
        DirectSrcVisitor<src_ctype> visitor;
        visitor.ptr = src;
        dispatch_with_visitor_quint8_dimshuffle_typecvt_nchw(
                is_nhwc, visitor, mat, dst, N_MAT, C, IH, IW, OH, OW, bval,
                dtype_param, bmode, stream);
    }
    after_kernel_launch();
}

#define INST(src_dtype, src_ctype, dst_ctype)                         \
    template void forward_proxy_quint8_dimshuffle_typecvt_nchw(       \
            bool is_nhwc, const src_ctype*, const float*, const int*, \
            dst_ctype*, int, int, int, int, int, int, int, src_ctype, \
            DTypeParamImpl<src_dtype> param, BorderMode,              \
            megcore::AsyncErrorInfo*, void*, cudaStream_t);

INST(dt_quint8, uint8_t, float)
#undef INST

}  // namespace warp_perspective
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
