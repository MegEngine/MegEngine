/**
 * \file dnn/src/cuda/relayout_format/relayout_format.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
    CudaPostProcess(){};
    CudaPostProcess(float, uint8_t, float, uint8_t){};
    inline __device__ int8_t operator()(int8_t val) { return val; }
};

template <>
struct CudaPostProcess<dtype::QuantizedS32, dtype::QuantizedS32, false> {
    CudaDTypeParamImpl<dt_qint32> m_dst_type_cvt;
    CudaDTypeParamImpl<dt_qint32> m_src_type_cvt;
    CudaPostProcess(float src_scale, int, float dst_scale, int) {
        m_dst_type_cvt = CudaDTypeParamImpl<dt_qint32>(dst_scale);
        m_src_type_cvt = CudaDTypeParamImpl<dt_qint32>(src_scale);
    };
    inline __device__ int operator()(int val) {
        float med_var = m_src_type_cvt.dequantize(dt_qint32(val));
        return m_dst_type_cvt.quantize(med_var).as_int32();
    }
};
template <>
struct CudaPostProcess<dtype::QuantizedS32, dtype::QuantizedS32, true> {
    CudaPostProcess(float, int, float, int){};
    inline __device__ int operator()(int val) { return val; }
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

template <>
struct DTypeRWHelper<int, 1> {
    using InnerDtype = int;
    using DstDtype = int4;
};

template <>
struct DTypeRWHelper<int, 4> {
    using InnerDtype = int4;
    using DstDtype = int4;
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

template <>
inline __device__ int4 make_zero_pad<int4>(const char zero_point) {
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

template <bool with_pad, int pack_w, int pack_c, bool same_scale, bool all_pad,
          typename SrcType, typename DstType, typename DnnSrcType,
          typename DnnDstType>
struct RelayoutKern {
    using InnerDtype = typename DTypeRWHelper<SrcType, pack_w>::InnerDtype;
    using DstDtype = typename DTypeRWHelper<SrcType, pack_w>::DstDtype;
    static inline __device__ void write(DstType* dst_ptr,
                                        DstDtype (&dst_width)[pack_w]) {
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

    static inline __device__ void fake_read(const SrcType* src_ptr,
                                            InnerDtype (&read_channel)[pack_c],
                                            const int ic_stride,
                                            const int remain_ic,
                                            const InnerDtype zero_point) {
#pragma unroll
        for (int ic_idx = 0; ic_idx < pack_c; ++ic_idx) {
            read_channel[ic_idx] = zero_point;
        }
    }

    static inline __device__ void core_relayout_kern(
            const SrcType* src, DstType* dst, const int src_offset_base,
            const int dst_offset_base, const int ic_offset, const int ic_stride,
            const int remain_ic,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        InnerDtype read_channel[pack_c];
        if (all_pad) {
            const InnerDtype zero_pad = make_zero_pad<InnerDtype>(zero_point);
            fake_read(src + ic_offset + src_offset_base, read_channel,
                      ic_stride, remain_ic, zero_pad);
        } else {
            if (with_pad) {
                const InnerDtype zero_pad =
                        make_zero_pad<InnerDtype>(zero_point);
                read_with_pad(src + ic_offset + src_offset_base, read_channel,
                              ic_stride, remain_ic, zero_pad);
            } else {
                read(src + ic_offset + src_offset_base, read_channel,
                     ic_stride);
            }
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
        const SrcType* src, DstType* dst, int in_n, int ic, int ihw,
        int n_stride_src, int ic_stride, int n_stride_dst,
        CudaPostProcess<DnnSrcType, DnnDstType, same_scale> post_process,
        const char zero_point, const int group, const int ocpg) {
    constexpr int pack_c = 4;
    const int n_idx = blockIdx.y;
    const int ihw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ihw_offset = ihw_block_idx * pack_w;

    if (ihw_offset < ihw) {
        const int src_offset_base = n_idx * n_stride_src + ihw_offset;
        const int dst_offset_base = n_idx * n_stride_dst + ihw_offset * pack_c;
        if (n_idx < in_n) {
            const int icpg = ic / group;
            const int ic_block = icpg / pack_c;
            const int remain_ic = icpg % pack_c;
            const int src_group_stride = icpg * ic_stride;
            const int dst_group_stride = ocpg * ic_stride;
            for (int g_idx = 0; g_idx < group; ++g_idx) {
                const int src_offset =
                        src_offset_base + g_idx * src_group_stride;
                const int dst_offset =
                        dst_offset_base + g_idx * dst_group_stride;
                for (int ic_blk_idx = 0; ic_blk_idx < ic_block; ++ic_blk_idx) {
                    const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                    RelayoutKern<false, pack_w, pack_c, same_scale, false,
                                 SrcType, DstType, DnnSrcType,
                                 DnnDstType>::core_relayout_kern(src, dst,
                                                                 src_offset,
                                                                 dst_offset,
                                                                 ic_offset,
                                                                 ic_stride,
                                                                 remain_ic,
                                                                 post_process,
                                                                 zero_point);
                }

                if (remain_ic > 0) {
                    const int ic_offset = ic_block * pack_c * ic_stride;
                    RelayoutKern<true, pack_w, pack_c, same_scale, false,
                                 SrcType, DstType, DnnSrcType,
                                 DnnDstType>::core_relayout_kern(src, dst,
                                                                 src_offset,
                                                                 dst_offset,
                                                                 ic_offset,
                                                                 ic_stride,
                                                                 remain_ic,
                                                                 post_process,
                                                                 zero_point);
                }
            }
        } else {
            //! pad n
            const int ic_full_block = group * ocpg / pack_c;
            for (int ic_blk_idx = 0; ic_blk_idx < ic_full_block; ++ic_blk_idx) {
                RelayoutKern<false, pack_w, pack_c, same_scale, true, SrcType,
                             DstType, DnnSrcType,
                             DnnDstType>::core_relayout_kern(src, dst,
                                                             src_offset_base,
                                                             dst_offset_base, 0,
                                                             ic_stride, 0,
                                                             post_process,
                                                             zero_point);
            }
        }
    }
}

__global__ void kern_nchw4_nchw(const int8_t* src, int8_t* dst, int n, int ic,
                                int oc, int oh, int ow, int group) {
    constexpr int pack_w = 1;
    constexpr int pack_ic = 4;
    const int n_idx = blockIdx.y;
    const int hw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hw_offset = hw_block_idx * pack_w;
    const int hw = oh * ow;
    const int n_stride_src = ic * hw;
    const int n_stride_dst = oc * hw;
    const int c_stride = hw;

    if (hw_offset < hw) {
        const int icpg = ic / group;
        const int ocpg = oc / group;
        const int src_group_stride = icpg * c_stride;
        const int dst_group_stride = ocpg * c_stride;
        for (int g_idx = 0; g_idx < group; ++g_idx) {
            const int oc_block = ocpg / pack_ic;
            const int remain_oc = ocpg % pack_ic;
            const int src_offset_base = n_idx * n_stride_src +
                                        g_idx * src_group_stride +
                                        hw_offset * pack_ic;
            const int dst_offset_base =
                    n_idx * n_stride_dst + g_idx * dst_group_stride + hw_offset;

            for (int ic_blk_idx = 0; ic_blk_idx < oc_block; ++ic_blk_idx) {
                const int oc_offset = ic_blk_idx * pack_ic * c_stride;
                char4 temp = *(char4*)(src + src_offset_base + oc_offset);
                dst[dst_offset_base + oc_offset + 0 * c_stride] = temp.x;
                dst[dst_offset_base + oc_offset + 1 * c_stride] = temp.y;
                dst[dst_offset_base + oc_offset + 2 * c_stride] = temp.z;
                dst[dst_offset_base + oc_offset + 3 * c_stride] = temp.w;
            }

            if (remain_oc > 0) {
                const int oc_offset = oc_block * pack_ic * c_stride;
                char4 temp = *(char4*)(src + src_offset_base + oc_offset);
                dst[dst_offset_base + oc_offset + 0 * c_stride] = temp.x;
                if (remain_oc > 1) {
                    dst[dst_offset_base + oc_offset + 1 * c_stride] = temp.y;
                }
                if (remain_oc > 2) {
                    dst[dst_offset_base + oc_offset + 2 * c_stride] = temp.z;
                }
            }
        }
    }
}

__global__ void kern_nchw_nchw4_weight(
        const char* src, char* dst, int in_oc, int ic, int ihw,
        int oc_stride_src, int ic_stride, int oc_stride_dst,
        int group_stride_src, int group_stride_dst, const char zero_point,
        CudaPostProcess<dtype::QuantizedS8, dtype::QuantizedS8, true>
                post_process) {
    typedef char SrcType;
    typedef char DstType;
    typedef dtype::QuantizedS8 DnnSrcType;
    typedef dtype::QuantizedS8 DnnDstType;
    constexpr int pack_c = 4;
    constexpr int pack_w = 1;
    constexpr bool same_scale = true;

    const int group_idx = blockIdx.z;
    const int oc_idx = blockIdx.y;
    const int ihw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ihw_offset = ihw_block_idx * pack_w;

    if (ihw_offset < ihw) {
        const int ic_block = ic / pack_c;
        const int remain_ic = ic % pack_c;
        const int src_offset_base = group_idx * group_stride_src +
                                    oc_idx * oc_stride_src + ihw_offset;
        const int dst_offset_base = group_idx * group_stride_dst +
                                    oc_idx * oc_stride_dst +
                                    ihw_offset * pack_c;
        if (oc_idx < in_oc) {
            for (int ic_blk_idx = 0; ic_blk_idx < ic_block; ++ic_blk_idx) {
                const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                RelayoutKern<false, pack_w, pack_c, same_scale, false, SrcType,
                             DstType, DnnSrcType,
                             DnnDstType>::core_relayout_kern(src, dst,
                                                             src_offset_base,
                                                             dst_offset_base,
                                                             ic_offset,
                                                             ic_stride,
                                                             remain_ic,
                                                             post_process,
                                                             zero_point);
            }

            if (remain_ic > 0) {
                const int ic_offset = ic_block * pack_c * ic_stride;
                RelayoutKern<true, pack_w, pack_c, same_scale, false, SrcType,
                             DstType, DnnSrcType,
                             DnnDstType>::core_relayout_kern(src, dst,
                                                             src_offset_base,
                                                             dst_offset_base,
                                                             ic_offset,
                                                             ic_stride,
                                                             remain_ic,
                                                             post_process,
                                                             zero_point);
            }
        } else {
            //! pad oc per group
            const int ic_full_block = (ic + pack_c - 1) / pack_c;
            for (int ic_blk_idx = 0; ic_blk_idx < ic_full_block; ++ic_blk_idx) {
                const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                RelayoutKern<false, pack_w, pack_c, same_scale, true, SrcType,
                             DstType, DnnSrcType,
                             DnnDstType>::core_relayout_kern(src, dst,
                                                             src_offset_base,
                                                             dst_offset_base,
                                                             ic_offset,
                                                             ic_stride,
                                                             remain_ic,
                                                             post_process,
                                                             zero_point);
            }
        }
    }
}

}  // namespace

template <int pack_w = 1>
void relayout_format::relayout_format_cuda_nchw_nchw4(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point,
        const int group) {
    constexpr int pack_oc = 4;
    const int in_n = src.layout[0];
    const int out_n = dst.layout[0];
    const int ic = src.layout[1];
    const int h = src.layout[2];
    const int w = src.layout[3];
    const int oc = dst.layout[1] * pack_oc;
    const int hw = h * w;
    const int ocpg = oc / group;
    const int n_stride_src = ic * hw;
    const int ic_stride = hw;
    const int n_stride_dst = oc * hw;

    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;
    bool same_scale = src_scale == dst_scale;
#define RUN_KERNEL(same_scale, SRC_TYPE, DST_TYPE, SRC_C_TYPE, DST_C_TYPE)     \
    if (same_scale) {                                                          \
        int nr_threads = query_blocksize_for_kernel(                           \
                kern_nchw_nchw4<pack_w, true, SRC_C_TYPE, DST_C_TYPE,          \
                                SRC_TYPE, DST_TYPE>);                          \
        const dim3 block_dim(DIVUP(hw, nr_threads* pack_w), out_n);            \
        const dim3 thread_dim(nr_threads);                                     \
        kern_nchw_nchw4<pack_w, true><<<block_dim, thread_dim, 0, stream>>>(   \
                (SRC_C_TYPE*)src.raw_ptr, (DST_C_TYPE*)dst.raw_ptr, in_n, ic,  \
                hw, n_stride_src, ic_stride, n_stride_dst,                     \
                CudaPostProcess<SRC_TYPE, DST_TYPE, true>(                     \
                        src_scale, src_zero_point, dst_scale, dst_zero_point), \
                src_zero_point, group, ocpg);                                  \
    } else {                                                                   \
        int nr_threads = query_blocksize_for_kernel(                           \
                kern_nchw_nchw4<pack_w, false, SRC_C_TYPE, DST_C_TYPE,         \
                                SRC_TYPE, DST_TYPE>);                          \
        const dim3 block_dim(DIVUP(hw, nr_threads* pack_w), out_n);            \
        const dim3 thread_dim(nr_threads);                                     \
        kern_nchw_nchw4<pack_w, false><<<block_dim, thread_dim, 0, stream>>>(  \
                (SRC_C_TYPE*)src.raw_ptr, (DST_C_TYPE*)dst.raw_ptr, in_n, ic,  \
                hw, n_stride_src, ic_stride, n_stride_dst,                     \
                CudaPostProcess<SRC_TYPE, DST_TYPE, false>(                    \
                        src_scale, src_zero_point, dst_scale, dst_zero_point), \
                src_zero_point, group, ocpg);                                  \
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
    } else if (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32 &&
               dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32) {
        RUN_KERNEL(same_scale, dtype::QuantizedS32, dtype::QuantizedS32, int,
                   int);
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
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32);
    return is_all_continue && is_all_int8;
}

void relayout_format::relayout_format_cuda_nchw4_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const int group) {
    constexpr int pack_w = 1;
    const int n = src.layout[0];
    const int ic = src.layout[1] * 4;
    const int h = src.layout[2];
    const int w = src.layout[3];
    const int oc = dst.layout[1];
    const int hw = h * w;
    int nr_threads = query_blocksize_for_kernel(kern_nchw4_nchw);
    const dim3 block_dim(DIVUP(hw, nr_threads * pack_w), n);
    const dim3 thread_dim(nr_threads);
    kern_nchw4_nchw<<<block_dim, thread_dim, 0, stream>>>(
            (int8_t*)src.raw_ptr, (int8_t*)dst.raw_ptr, n, ic, oc, h, w, group);
}

void relayout_format::relayout_format_cuda_nchw_nchw4_weight(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream) {
    constexpr int pack_c = 4;
    const bool is_group = src.layout.ndim == 5;
    const int group = is_group ? src.layout[0] : 1;
    const int oc = is_group ? src.layout[1] : src.layout[0];
    const int ic = is_group ? src.layout[2] : src.layout[1];
    const int kh = is_group ? src.layout[3] : src.layout[2];
    const int kw = is_group ? src.layout[4] : src.layout[3];
    const int hw = kh * kw;
    const int oc_round = ROUNDUP(oc, pack_c);
    const int ic_round = ROUNDUP(ic, pack_c);
    const int ic_stride = hw;
    const int oc_stride_src = ic * ic_stride;
    const int oc_stride_dst = ic_round * ic_stride;
    const int group_stride_src = oc * oc_stride_src;
    const int group_stride_dst = oc_round * oc_stride_dst;

    int nr_threads = 32;
    const dim3 block_dim(DIVUP(hw, nr_threads), oc_round, group);
    const dim3 thread_dim(nr_threads);

    kern_nchw_nchw4_weight<<<block_dim, thread_dim, 0, stream>>>(
            (char*)src.raw_ptr, (char*)dst.raw_ptr, oc, ic, hw, oc_stride_src,
            ic_stride, oc_stride_dst, group_stride_src, group_stride_dst, 0,
            {});
}

template void relayout_format::relayout_format_cuda_nchw_nchw4<1>(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point,
        const int group);

template void relayout_format::relayout_format_cuda_nchw_nchw4<4>(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point,
        const int group);
