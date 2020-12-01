/**
 * \file dnn/src/cuda/relayout_format/relayout_format.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/relayout_format/relayout_format.cuh"
using namespace megdnn;
using namespace cuda;

namespace {

template <typename SrcType, typename DstType, bool same_scale>
struct CudaPostProcess;

template <>
struct CudaPostProcess<dtype::Uint8, dtype::QuantizedS8, true> {
    CudaPostProcess(float, uint8_t, float, uint8_t){};
    inline __device__ int8_t operator()(uint8_t val) { return val - 128; }
};

template <>
struct CudaPostProcess<dtype::Uint8, dtype::QuantizedS8, false> {
    CudaDTypeParamImpl<dt_qint8> m_dst_type_cvt;
    CudaPostProcess(float, uint8_t, float dst_scale, uint8_t) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint8>(dst_scale);
    };
    inline __device__ int8_t operator()(uint8_t val) {
        return m_dst_type_cvt.quantize((float)val - 128.f).as_int8();
    }
};

template <>
struct CudaPostProcess<dtype::Quantized8Asymm, dtype::QuantizedS8, false> {
    CudaDTypeParamImpl<dt_qint8> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_quint8> m_src_type_cvt;
    CudaPostProcess(float src_scale, uint8_t src_zero_point, float dst_scale,
                    uint8_t) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint8>(dst_scale);
        m_src_type_cvt =
                CudaDTypeParamImpl<dt_quint8>(src_scale, src_zero_point);
    };
    inline __device__ int8_t operator()(uint8_t val) {
        float med_var = m_src_type_cvt.dequantize(dt_quint8(val));
        return m_dst_type_cvt.quantize(med_var).as_int8();
    }
};

template <>
struct CudaPostProcess<dtype::Quantized8Asymm, dtype::QuantizedS8, true> {
    uint8_t m_src_zero_point = 0;
    CudaPostProcess(float, uint8_t src_zero_point, float, uint8_t) {
        m_src_zero_point = src_zero_point;
    };
    inline __device__ int8_t operator()(uint8_t val) {
        return val - m_src_zero_point;
    }
};

template <>
struct CudaPostProcess<dtype::QuantizedS8, dtype::QuantizedS8, false> {
    CudaDTypeParamImpl<dt_qint8> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_qint8> m_src_type_cvt;
    CudaPostProcess(float src_scale, uint8_t, float dst_scale, uint8_t) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint8>(dst_scale);
        m_src_type_cvt = CudaDTypeParamImpl<dt_qint8>(src_scale);
    };
    inline __device__ int8_t operator()(int8_t val) {
        float med_var = m_src_type_cvt.dequantize(dt_qint8(val));
        return m_dst_type_cvt.quantize(med_var).as_int8();
    }
};

template <>
struct CudaPostProcess<dtype::QuantizedS8, dtype::QuantizedS8, true> {
    CudaPostProcess(float, uint8_t, float, uint8_t){};
    inline __device__ int8_t operator()(int8_t val) { return val; }
};

template <typename SrcType, int pack_w>
struct DTypeRWHelper;
template <>
struct DTypeRWHelper<char, 1> {
    using InnerDtype = char;
    using DstDtype = char4;
};

template <>
struct DTypeRWHelper<char, 4> {
    using InnerDtype = char4;
    using DstDtype = char4;
};

template <int pack_w, int pack_c, typename SrcType, typename DnnSrcType,
          typename DnnDstType, bool same_scale>
struct Translayout {
    using InnerDtype = typename DTypeRWHelper<SrcType, pack_w>::InnerDtype;
    using DstDtype = typename DTypeRWHelper<SrcType, pack_w>::DstDtype;
    static inline __device__ void trans(DstDtype (&dst_width)[pack_w],
                                        InnerDtype (&read_channel)[pack_c],
                                        const char zero_point);
};

template <typename SrcType, typename DnnSrcType, typename DnnDstType,
          bool same_scale>
struct Translayout<1, 4, SrcType, DnnSrcType, DnnDstType, same_scale> {
    using InnerDtype = typename DTypeRWHelper<SrcType, 1>::InnerDtype;
    using DstDtype = typename DTypeRWHelper<SrcType, 1>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[1], InnerDtype (&read_channel)[4],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        dst_width[0].x = post_process(read_channel[0]);
        dst_width[0].y = post_process(read_channel[1]);
        dst_width[0].z = post_process(read_channel[2]);
        dst_width[0].w = post_process(read_channel[3]);
    }
};

template <typename SrcType, typename DnnSrcType, typename DnnDstType,
          bool same_scale>
struct Translayout<4, 4, SrcType, DnnSrcType, DnnDstType, same_scale> {
    using InnerDtype = typename DTypeRWHelper<SrcType, 4>::InnerDtype;
    using DstDtype = typename DTypeRWHelper<SrcType, 4>::DstDtype;
    static inline __device__ void trans(
            DstDtype (&dst_width)[4], InnerDtype (&read_channel)[4],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        dst_width[0].x = post_process(read_channel[0].x);
        dst_width[0].y = post_process(read_channel[1].x);
        dst_width[0].z = post_process(read_channel[2].x);
        dst_width[0].w = post_process(read_channel[3].x);

        dst_width[1].x = post_process(read_channel[0].y);
        dst_width[1].y = post_process(read_channel[1].y);
        dst_width[1].z = post_process(read_channel[2].y);
        dst_width[1].w = post_process(read_channel[3].y);

        dst_width[2].x = post_process(read_channel[0].z);
        dst_width[2].y = post_process(read_channel[1].z);
        dst_width[2].z = post_process(read_channel[2].z);
        dst_width[2].w = post_process(read_channel[3].z);

        dst_width[3].x = post_process(read_channel[0].w);
        dst_width[3].y = post_process(read_channel[1].w);
        dst_width[3].z = post_process(read_channel[2].w);
        dst_width[3].w = post_process(read_channel[3].w);
    }
};

template <typename DstType>
inline __device__ DstType make_zero_pad(const char zero_point) {
    return zero_point;
}

template <>
inline __device__ char4 make_zero_pad<char4>(const char zero_point) {
    return {zero_point, zero_point, zero_point, zero_point};
}

template <typename DstDtype>
inline __device__ void write_helper(DstDtype* ptr, DstDtype val) {
    *ptr = val;
}

template <>
inline __device__ void write_helper<char4>(char4* ptr, char4 val) {
    int32_t* rel_ptr = (int32_t*)ptr;
    *rel_ptr = *(int32_t*)(&val);
}

template <bool with_pad, int pack_w, int pack_c, bool same_scale,
          typename SrcType, typename DstType, typename DnnSrcType,
          typename DnnDstType>
struct RelayoutKern {
    using InnerDtype = typename DTypeRWHelper<SrcType, pack_w>::InnerDtype;
    using DstDtype = typename DTypeRWHelper<SrcType, pack_w>::DstDtype;
    static inline __device__ void write(DstType* dst_ptr,
                                        char4 (&dst_width)[pack_w]) {
        DstDtype* dst_inner_ptr = (DstDtype*)dst_ptr;
#pragma unroll
        for (int iw_idx = 0; iw_idx < pack_w; ++iw_idx) {
            write_helper(dst_inner_ptr + iw_idx, dst_width[iw_idx]);
        }
    }

    static inline __device__ void read(const SrcType* src_ptr,
                                       InnerDtype (&read_channel)[pack_c],
                                       const int ic_stride) {
#pragma unroll
        for (int ic_idx = 0; ic_idx < pack_c; ++ic_idx) {
            read_channel[ic_idx] = *(InnerDtype*)(src_ptr + ic_idx * ic_stride);
        }
    }

    static inline __device__ void read_with_pad(
            const SrcType* src_ptr, InnerDtype (&read_channel)[pack_c],
            const int ic_stride, const int remain_ic,
            const InnerDtype zero_point) {
#pragma unroll
        for (int ic_idx = 0; ic_idx < pack_c; ++ic_idx) {
            read_channel[ic_idx] =
                    ic_idx < remain_ic
                            ? *(InnerDtype*)(src_ptr + ic_idx * ic_stride)
                            : zero_point;
        }
    }

    static inline __device__ void core_relayout_kern(
            const SrcType* src, DstType* dst, const int src_offset_base,
            const int dst_offset_base, const int ic_offset, const int ic_stride,
            const int remain_ic,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        InnerDtype read_channel[pack_c];
        if (with_pad) {
            const InnerDtype zero_pad = make_zero_pad<InnerDtype>(zero_point);
            read_with_pad(src + ic_offset + src_offset_base, read_channel,
                          ic_stride, remain_ic, zero_pad);
        } else {
            read(src + ic_offset + src_offset_base, read_channel, ic_stride);
        }
        DstDtype dst_width[pack_w];
        Translayout<pack_w, pack_c, SrcType, DnnSrcType, DnnDstType,
                    same_scale>::trans(dst_width, read_channel, post_process,
                                       zero_point);
        write(dst + ic_offset + dst_offset_base, dst_width);
    }
};

template <int pack_w, bool same_scale, typename SrcType, typename DstType,
          typename DnnSrcType, typename DnnDstType>
__global__ void kern_nchw_nchw4(
        const SrcType* src, DstType* dst, int ic, int ihw, int n_stride_src,
        int ic_stride, int n_stride_dst,
        CudaPostProcess<DnnSrcType, DnnDstType, same_scale> post_process,
        const char zero_point) {
    constexpr int pack_c = 4;
    const int n_idx = blockIdx.y;
    const int ihw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ihw_offset = ihw_block_idx * pack_w;

    if (ihw_offset < ihw) {
        const int ic_block = ic / pack_c;
        const int remain_ic = ic % pack_c;
        const int src_offset_base = n_idx * n_stride_src + ihw_offset;
        const int dst_offset_base = n_idx * n_stride_dst + ihw_offset * pack_c;

        for (int ic_blk_idx = 0; ic_blk_idx < ic_block; ++ic_blk_idx) {
            const int ic_offset = ic_blk_idx * pack_c * ic_stride;
            RelayoutKern<false, pack_w, pack_c, same_scale, SrcType, DstType,
                         DnnSrcType,
                         DnnDstType>::core_relayout_kern(src, dst,
                                                         src_offset_base,
                                                         dst_offset_base,
                                                         ic_offset, ic_stride,
                                                         remain_ic,
                                                         post_process,
                                                         zero_point);
        }

        if (remain_ic > 0) {
            const int ic_offset = ic_block * pack_c * ic_stride;
            RelayoutKern<true, pack_w, pack_c, same_scale, SrcType, DstType,
                         DnnSrcType,
                         DnnDstType>::core_relayout_kern(src, dst,
                                                         src_offset_base,
                                                         dst_offset_base,
                                                         ic_offset, ic_stride,
                                                         remain_ic,
                                                         post_process,
                                                         zero_point);
        }
    }
}

}  // namespace

template <int pack_w = 1>
void relayout_format::relayout_format_cuda_exec(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point) {
    constexpr int pack_oc = 4;
    const int n = src.layout[0];
    const int c = src.layout[1];
    const int h = src.layout[2];
    const int w = src.layout[3];
    const int hw = h * w;
    const int oc_block = DIVUP(c, pack_oc);
    const int n_stride_src = c * hw;
    const int ic_stride = hw;
    const int n_stride_dst = oc_block * pack_oc * h * w;

    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;
    bool same_scale = src_scale == dst_scale;
#define RUN_KERNEL(same_scale, SRC_TYPE, DST_TYPE, SRC_C_TYPE, DST_C_TYPE)     \
    if (same_scale) {                                                          \
        int nr_threads = query_blocksize_for_kernel(                           \
                kern_nchw_nchw4<pack_w, true, SRC_C_TYPE, DST_C_TYPE,          \
                                SRC_TYPE, DST_TYPE>);                          \
        const dim3 block_dim(DIVUP(hw, nr_threads* pack_w), n);                \
        const dim3 thread_dim(nr_threads);                                     \
        kern_nchw_nchw4<pack_w, true><<<block_dim, thread_dim, 0, stream>>>(   \
                (SRC_C_TYPE*)src.raw_ptr, (DST_C_TYPE*)dst.raw_ptr, c, hw,     \
                n_stride_src, ic_stride, n_stride_dst,                         \
                CudaPostProcess<SRC_TYPE, DST_TYPE, true>(                     \
                        src_scale, src_zero_point, dst_scale, dst_zero_point), \
                src_zero_point);                                               \
    } else {                                                                   \
        int nr_threads = query_blocksize_for_kernel(                           \
                kern_nchw_nchw4<pack_w, false, SRC_C_TYPE, DST_C_TYPE,         \
                                SRC_TYPE, DST_TYPE>);                          \
        const dim3 block_dim(DIVUP(hw, nr_threads* pack_w), n);                \
        const dim3 thread_dim(nr_threads);                                     \
        kern_nchw_nchw4<pack_w, false><<<block_dim, thread_dim, 0, stream>>>(  \
                (SRC_C_TYPE*)src.raw_ptr, (DST_C_TYPE*)dst.raw_ptr, c, hw,     \
                n_stride_src, ic_stride, n_stride_dst,                         \
                CudaPostProcess<SRC_TYPE, DST_TYPE, false>(                    \
                        src_scale, src_zero_point, dst_scale, dst_zero_point), \
                src_zero_point);                                               \
    }

    if (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Uint8 &&
        dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) {
        RUN_KERNEL(same_scale, dtype::Uint8, dtype::QuantizedS8, char, char);
    } else if (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized8Asymm &&
               dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) {
        RUN_KERNEL(same_scale, dtype::Quantized8Asymm, dtype::QuantizedS8, char,
                   char);
    } else if (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8 &&
               dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) {
        RUN_KERNEL(same_scale, dtype::QuantizedS8, dtype::QuantizedS8, char,
                   char);
    } else {
        megdnn_assert(0, "not support dtype %s %s", src_layout.dtype.name(),
                      dst_layout.dtype.name());
    }
}

bool relayout_format::relayout_format_cuda_usable(
        const TensorLayout& src_layout, const TensorLayout& dst_layout) {
    bool is_all_continue =
            src_layout.is_contiguous() && dst_layout.is_contiguous();
    bool is_all_int8 =
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Uint8 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized8Asymm &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8);
    return is_all_continue && is_all_int8;
}

template void relayout_format::relayout_format_cuda_exec<1>(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point);

template void relayout_format::relayout_format_cuda_exec<4>(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point);
