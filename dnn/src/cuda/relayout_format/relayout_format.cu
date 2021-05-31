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
#include "src/cuda/relayout_format/relayout_format_kern.cuh"

using namespace megdnn;
using namespace cuda;
using namespace relayout_format;
using namespace internal;

namespace {
template <bool with_pad, int pack_w, int pack_c, bool same_scale, bool all_pad,
          typename SrcType, typename DstType, typename DnnSrcType,
          typename DnnDstType>
struct RelayoutKern {
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   pack_w>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   pack_w>::DstDtype;
    static inline __device__ void write(DstDtype* dst_ptr,
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
            const SrcType* src, DstType* dst, const int ic_stride,
            const int remain_ic,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const uint8_t zero_point) {
        InnerDtype read_channel[pack_c];
        if (all_pad) {
            const InnerDtype zero_pad = make_zero_pad<InnerDtype>(zero_point);
            fake_read(src, read_channel, ic_stride, remain_ic, zero_pad);
        } else {
            if (with_pad) {
                const InnerDtype zero_pad =
                        make_zero_pad<InnerDtype>(zero_point);
                read_with_pad(src, read_channel, ic_stride, remain_ic,
                              zero_pad);
            } else {
                read(src, read_channel, ic_stride);
            }
        }
        DstDtype dst_width[pack_w];
        Translayout<pack_w, pack_c, SrcType, DnnSrcType, DnnDstType,
                    same_scale>::trans(dst_width, read_channel, post_process,
                                       zero_point);
        write(reinterpret_cast<DstDtype*>(dst), dst_width);
    }
};

template <int pack_w, int pack_c, bool same_scale, typename SrcType,
          typename DstType, typename DnnSrcType, typename DnnDstType,
          int size_nbits = 8>
__global__ void kern_nchw_nchwx(
        const SrcType* src, DstType* dst, int in_n, int ic, int ihw,
        int n_stride_src, int ic_stride, int n_stride_dst, int oc_stride,
        CudaPostProcess<DnnSrcType, DnnDstType, same_scale> post_process,
        const uint8_t zero_point, const int group, const int ocpg) {
    static constexpr int size_src_type = sizeof(SrcType);
    static constexpr int size_dst_type = sizeof(DstType);
#ifndef MEGDNN_COMMA
#define MEGDNN_COMMA ,
#endif
    MEGDNN_STATIC_ASSERT(std::is_same<SrcType MEGDNN_COMMA DstType>::value,
                         "Currently this kernel only support accessing tensor "
                         "src and dst in same data type.");
    n_stride_src /= size_src_type;
    ic_stride /= size_src_type;
    n_stride_dst /= size_dst_type;
    oc_stride /= size_dst_type;

    const int n_idx = blockIdx.y;
    const int ihw_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ihw_offset =
            ihw_block_idx * pack_w;
    const int ihw_offset_in_type =
            ihw_offset * size_nbits / (8 * size_src_type);
    if (ihw_offset < ihw) {
        const int src_offset_base = n_idx * n_stride_src + ihw_offset_in_type;
        const int dst_offset_base =
                n_idx * n_stride_dst + ihw_offset_in_type * pack_c;
        if (n_idx < in_n) {
            const int icpg = ic / group;
            const int ic_block = icpg / pack_c;
            const int remain_ic = icpg % pack_c;
            const int src_group_stride = icpg * ic_stride;
            const int dst_group_stride = (ocpg / pack_c) * oc_stride;
            for (int g_idx = 0; g_idx < group; ++g_idx) {
                const int src_offset =
                        src_offset_base + g_idx * src_group_stride;
                const int dst_offset =
                        dst_offset_base + g_idx * dst_group_stride;
                for (int ic_blk_idx = 0; ic_blk_idx < ic_block; ++ic_blk_idx) {
                    const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                    const int oc_offset = ic_blk_idx * oc_stride;
                    RelayoutKern<false, pack_w, pack_c, same_scale, false,
                                 SrcType, DstType, DnnSrcType, DnnDstType>::
                            core_relayout_kern(src + src_offset + ic_offset,
                                               dst + dst_offset + oc_offset,
                                               ic_stride, remain_ic,
                                               post_process, zero_point);
                }

                if (remain_ic > 0) {
                    const int ic_offset = ic_block * pack_c * ic_stride;
                    const int oc_offset = ic_block * oc_stride;
                    RelayoutKern<true, pack_w, pack_c, same_scale, false,
                                 SrcType, DstType, DnnSrcType, DnnDstType>::
                            core_relayout_kern(src + src_offset + ic_offset,
                                               dst + dst_offset + oc_offset,
                                               ic_stride, remain_ic,
                                               post_process, zero_point);
                }
            }
        } else {
            //! pad n
            const int ic_full_block = group * ocpg / pack_c;
            for (int ic_blk_idx = 0; ic_blk_idx < ic_full_block; ++ic_blk_idx) {
                RelayoutKern<false, pack_w, pack_c, same_scale, true, SrcType,
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base,
                                           dst + dst_offset_base, ic_stride, 0,
                                           post_process, zero_point);
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
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base + ic_offset,
                                           dst + dst_offset_base + ic_offset,
                                           ic_stride, remain_ic, post_process,
                                           zero_point);
            }

            if (remain_ic > 0) {
                const int ic_offset = ic_block * pack_c * ic_stride;
                RelayoutKern<true, pack_w, pack_c, same_scale, false, SrcType,
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base + ic_offset,
                                           dst + dst_offset_base + ic_offset,
                                           ic_stride, remain_ic, post_process,
                                           zero_point);
            }
        } else {
            //! pad oc per group
            const int ic_full_block = (ic + pack_c - 1) / pack_c;
            for (int ic_blk_idx = 0; ic_blk_idx < ic_full_block; ++ic_blk_idx) {
                const int ic_offset = ic_blk_idx * pack_c * ic_stride;
                RelayoutKern<false, pack_w, pack_c, same_scale, true, SrcType,
                             DstType, DnnSrcType, DnnDstType>::
                        core_relayout_kern(src + src_offset_base + ic_offset,
                                           dst + dst_offset_base + ic_offset,
                                           ic_stride, remain_ic, post_process,
                                           zero_point);
            }
        }
    }
}
}  // namespace

void relayout_format::relayout_format_cuda_nchw_nchwx(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point, int group) {
    auto&& stype = src.layout.dtype;
    auto&& dtype = dst.layout.dtype;
    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;
    // check pack size
    int pack_oc = std::numeric_limits<int>::min();
#define DEF(_pack_oc, _src_type, _dst_type)             \
    if (stype.enumv().ev == DTypeEnum::Ev::_src_type && \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) { \
        pack_oc = _pack_oc;                             \
    }
    // clang-format off
    DEF(64, QuantizedS4, QuantizedS4)
    DEF(64, Quantized4Asymm, Quantized4Asymm)
    DEF(4, QuantizedS8, QuantizedS8)
    DEF(4, Uint8, QuantizedS8)
    DEF(4, Quantized8Asymm, QuantizedS8)
    DEF(4, QuantizedS32, QuantizedS32)
    // clang-format on
    megdnn_assert(pack_oc == 4 || pack_oc == 64,
                  "Unsupport pack size(pack_oc:%d, src:%s, dst:%s)", pack_oc,
                  stype.name(), dtype.name());
#undef DEF
    // no padding
    if (stype.enumv().ev != DTypeEnum::Ev::QuantizedS4 &&
        stype.enumv().ev != DTypeEnum::Ev::Quantized4Asymm) {
        const int in_n = src.layout[0];
        const int out_n = dst.layout[0];
        const int ic = src.layout[1];
        const int h = src.layout[2];
        const int w = src.layout[3];
        const int oc = dst.layout[1] * pack_oc;
        const int hw = h * w;
        const int ocpg = oc / group;
        // stride in byte
        const int n_stride_src = src_layout.dtype.size(src_layout.stride[0]);
        const int ic_stride = src_layout.dtype.size(src_layout.stride[1]);
        const int n_stride_dst = dst_layout.dtype.size(dst_layout.stride[0]);
        const int oc_stride = dst_layout.dtype.size(dst_layout.stride[1]);

        bool same_scale = src_scale == dst_scale;
#define DISPATCH_RAW(_same_scale, _pack_w, _pack_oc, _src_type, _dst_type,   \
                     _src_c_type, _dst_c_type, _size_nbits)                  \
    if (same_scale == _same_scale && hw % _pack_w == 0 &&                    \
        stype.enumv().ev == DTypeEnum::Ev::_src_type &&                      \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                      \
        auto kernel =                                                        \
                kern_nchw_nchwx<_pack_w, _pack_oc, _same_scale, _src_c_type, \
                                _dst_c_type, dtype::_src_type,               \
                                dtype::_dst_type, _size_nbits>;              \
        int nr_threads = query_blocksize_for_kernel(kernel);                 \
        const dim3 block_dim(DIVUP(hw, nr_threads* _pack_w), out_n);         \
        const dim3 thread_dim(nr_threads);                                   \
        return kernel<<<block_dim, thread_dim, 0, stream>>>(                 \
                (_src_c_type*)src.raw_ptr, (_dst_c_type*)dst.raw_ptr, in_n,  \
                ic, hw, n_stride_src, ic_stride, n_stride_dst, oc_stride,    \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,          \
                                _same_scale>(src_scale, src_zero_point,      \
                                             dst_scale, dst_zero_point),     \
                src_zero_point, group, ocpg);                                \
    }
#define DISPATCH_INT(_src_type, _dst_type)                         \
    DISPATCH_RAW(true, 4, 4, _src_type, _dst_type, int, int, 32);  \
    DISPATCH_RAW(false, 4, 4, _src_type, _dst_type, int, int, 32); \
    DISPATCH_RAW(true, 1, 4, _src_type, _dst_type, int, int, 32);  \
    DISPATCH_RAW(false, 1, 4, _src_type, _dst_type, int, int, 32);
#define DISPATCH_BYTE(_src_type, _dst_type)                         \
    DISPATCH_RAW(true, 4, 4, _src_type, _dst_type, char, char, 8);  \
    DISPATCH_RAW(false, 4, 4, _src_type, _dst_type, char, char, 8); \
    DISPATCH_RAW(true, 1, 4, _src_type, _dst_type, char, char, 8);  \
    DISPATCH_RAW(false, 1, 4, _src_type, _dst_type, char, char, 8);
        DISPATCH_INT(QuantizedS32, QuantizedS32);
        DISPATCH_BYTE(Uint8, QuantizedS8);
        DISPATCH_BYTE(Quantized8Asymm, QuantizedS8);
        DISPATCH_BYTE(QuantizedS8, QuantizedS8);
#undef DISPATCH_BYTE
#undef DISPATCH_INT
#undef DISPATCH_RAW
        megdnn_assert(
                false,
                "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                stype.name(), dtype.name(), h, w);
    } else {
        megdnn_assert(src_layout.dtype.is_low_bit());
        int n = src.layout[0];
        int ic = src.layout[1];
        int oc = dst.layout[1] * pack_oc;
        int h = src.layout[2];
        // align to byte
        int w = src.layout[3];
        int w_pad = DIVUP(w, 2) * 2;
        int hw  = h * w_pad;
        int n_stride_src = src_layout.stride[0];
        int ic_stride = src_layout.stride[1];
        int n_stride_dst = dst_layout.stride[0];
        int oc_stride = dst_layout.stride[1];
        int problem_size = n * (oc / pack_oc) * hw;
        bool same_scale = src_scale == dst_scale;
        bool padding = w % 2 != 0;
#define DISPATCH_RAW(_padding, _same_scale, _pack_w, _pack_oc, _src_type,      \
                     _dst_type, _src_c_type, _dst_c_type, _size_nbits)         \
    if (padding == _padding && same_scale == _same_scale &&                    \
        hw % _pack_w == 0 && stype.enumv().ev == DTypeEnum::Ev::_src_type &&   \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                        \
        using InnerDtype_ = typename DTypeRWHelper<                            \
                typename DTypeTrait<dtype::_src_type>::ctype,                  \
                _pack_w>::InnerDtype;                                          \
        using SrcIterator_ =                                                   \
                TensorIteratorOverChannel<InnerDtype_, 1, _pack_oc, _pack_w,   \
                                          _size_nbits>;                        \
        using DstIterator_ =                                                   \
                typename TensorIteratorPolicy<_padding, _dst_c_type, _pack_oc, \
                                              _pack_oc, _pack_w,               \
                                              _size_nbits>::TensorIterator;    \
        using CudaPostProcess_ =                                               \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,            \
                                _same_scale>;                                  \
        using Transpose_ =                                                     \
                Translayout<_pack_w, _pack_oc, _src_c_type, dtype::_src_type,  \
                            dtype::_dst_type, _same_scale>;                    \
        using RelayoutProblem_ =                                               \
                RelayoutProblem<SrcIterator_, DstIterator_, Transpose_,        \
                                CudaPostProcess_>;                             \
        n_stride_src = n_stride_src * _size_nbits / (8 * sizeof(InnerDtype_)); \
        ic_stride = ic_stride * _size_nbits / (8 * sizeof(InnerDtype_));       \
        n_stride_dst = n_stride_dst * _size_nbits / (8 * sizeof(_dst_c_type)); \
        oc_stride = oc_stride * _size_nbits / (8 * sizeof(_dst_c_type));       \
        typename RelayoutProblem_::Param param{                                \
                SrcIterator_{(InnerDtype_*)src.raw_ptr, ic_stride, ic, w,      \
                             w_pad},                                           \
                DstIterator_{(_dst_c_type*)dst.raw_ptr, oc_stride, oc, w,      \
                             w_pad},                                           \
                CudaPostProcess_{src_scale, src_zero_point, dst_scale,         \
                                 dst_zero_point},                              \
                n_stride_src,                                                  \
                n_stride_dst,                                                  \
                n,                                                             \
                oc,                                                            \
                hw,                                                            \
                src_zero_point};                                               \
        auto kernel = relayout_kern<RelayoutProblem_>;                         \
        int nr_threads = query_blocksize_for_kernel(kernel);                   \
        nr_threads = std::min(nr_threads, DIVUP(problem_size, _pack_w));       \
        const dim3 block_dim(DIVUP(problem_size, nr_threads* _pack_w));        \
        const dim3 thread_dim(nr_threads);                                     \
        return kernel<<<block_dim, thread_dim, 0, stream>>>(param);            \
    }
#define DISPATCH_4BITS(_src_type, _dst_type)                                \
    DISPATCH_RAW(true, true, 8, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(true, true, 2, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, true, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 8, 64, _src_type, _dst_type, char, char, 4); \
    DISPATCH_RAW(false, true, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 2, 64, _src_type, _dst_type, char, char, 4);
        DISPATCH_4BITS(QuantizedS4, QuantizedS4);
        DISPATCH_4BITS(Quantized4Asymm, Quantized4Asymm);
#undef DISPATCH_4BITS
#undef DISPATCH_RAW
        megdnn_assert(
                false,
                "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                stype.name(), dtype.name(), h, w);
    }
}

bool relayout_format::relayout_format_cuda_usable(
        const TensorLayout& src_layout, const TensorLayout& dst_layout) {
    bool is_all_continue =
            src_layout.is_contiguous() && dst_layout.is_contiguous();
    bool is_all_int32 =
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS32);
    bool is_all_int8 =
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Uint8 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized8Asymm &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS8);
    bool is_all_int4 =
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS4 &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::QuantizedS4) ||
            (src_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized4Asymm &&
             dst_layout.dtype.enumv().ev == DTypeEnum::Ev::Quantized4Asymm);
    return is_all_continue && (is_all_int32 || is_all_int8 || is_all_int4);
}

void relayout_format::relayout_format_cuda_nchwx_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point) {
    auto&& stype = src.layout.dtype;
    auto&& dtype = dst.layout.dtype;
    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;
    // check pack size
    int pack_ic = std::numeric_limits<int>::min();
#define DEF(_pack_ic, _src_type, _dst_type)             \
    if (stype.enumv().ev == DTypeEnum::Ev::_src_type && \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) { \
        pack_ic = _pack_ic;                             \
    }
    // clang-format off
    DEF(64, QuantizedS4, QuantizedS4)
    DEF(64, Quantized4Asymm, Quantized4Asymm)
    // clang-format on
    megdnn_assert(pack_ic == 64, "Unsupport pack size(pack_ic:%d)", pack_ic);
#undef DEF
    int n = src.layout[0];
    int ic = src.layout[1] * pack_ic;
    int h = src.layout[2];
    // align to byte
    int w = src.layout[3];
    int w_pad = DIVUP(w, 2) * 2;
    int hw = h * w_pad;
    int n_stride_src = src_layout.stride[0];
    int ic_stride = src_layout.stride[1];
    int n_stride_dst = dst_layout.stride[0];
    int oc_stride = dst_layout.stride[1];
    int problem_size = n * (ic / pack_ic) * hw;
    int oc = dst.layout[1];

    bool same_scale = src_scale == dst_scale;
    bool padding = w % 2 != 0;
#define DISPATCH_RAW(_padding, _same_scale, _pack_w, _pack_oc, _src_type,      \
                     _dst_type, _src_c_type, _dst_c_type, _size_nbits)         \
    if (padding == _padding && same_scale == _same_scale &&                    \
        hw % _pack_w == 0 && stype.enumv().ev == DTypeEnum::Ev::_src_type &&   \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                        \
        using SrcIterator_ =                                                   \
                typename TensorIteratorPolicy<_padding, _src_c_type, _pack_oc, \
                                              _pack_oc, _pack_w,               \
                                              _size_nbits>::TensorIterator;    \
        using InnerDtype_ = typename DTypeRWHelper<                            \
                typename DTypeTrait<dtype::_src_type>::ctype,                  \
                _pack_w>::InnerDtype;                                          \
        using DstIterator_ =                                                   \
                TensorIteratorOverChannel<InnerDtype_, 1, _pack_oc, _pack_w,   \
                                          _size_nbits>;                        \
        using CudaPostProcess_ =                                               \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,            \
                                _same_scale>;                                  \
        using Transpose_ =                                                     \
                Translayout<_pack_oc, _pack_w, _src_c_type, dtype::_src_type,  \
                            dtype::_dst_type, _same_scale>;                    \
        using RelayoutProblem_ =                                               \
                RelayoutProblem<SrcIterator_, DstIterator_, Transpose_,        \
                                CudaPostProcess_>;                             \
        n_stride_src = n_stride_src * _size_nbits / (8 * sizeof(_src_c_type)); \
        ic_stride = ic_stride * _size_nbits / (8 * sizeof(_src_c_type));       \
        n_stride_dst = n_stride_dst * _size_nbits / (8 * sizeof(InnerDtype_)); \
        oc_stride = oc_stride * _size_nbits / (8 * sizeof(InnerDtype_));       \
        typename RelayoutProblem_::Param param{                                \
                SrcIterator_{(_src_c_type*)src.raw_ptr, ic_stride, ic, w,      \
                             w_pad},                                           \
                DstIterator_{(InnerDtype_*)dst.raw_ptr, oc_stride, oc, w,      \
                             w_pad},                                           \
                CudaPostProcess_{src_scale, src_zero_point, dst_scale,         \
                                 dst_zero_point},                              \
                n_stride_src,                                                  \
                n_stride_dst,                                                  \
                n,                                                             \
                ic,                                                            \
                hw,                                                            \
                src_zero_point};                                               \
        auto kernel = relayout_kern<RelayoutProblem_>;                         \
        int nr_threads = query_blocksize_for_kernel(kernel);                   \
        nr_threads = std::min(nr_threads, DIVUP(problem_size, _pack_w));       \
        const dim3 block_dim(DIVUP(problem_size, nr_threads* _pack_w));        \
        const dim3 thread_dim(nr_threads);                                     \
        return kernel<<<block_dim, thread_dim, 0, stream>>>(param);            \
    }
#define DISPATCH_4BITS(_src_type, _dst_type)                                \
    DISPATCH_RAW(true, true, 8, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(true, true, 2, 64, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, true, 8, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 8, 64, _src_type, _dst_type, char, char, 4); \
    DISPATCH_RAW(false, true, 2, 64, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 2, 64, _src_type, _dst_type, char, char, 4);
    DISPATCH_4BITS(QuantizedS4, QuantizedS4);
    DISPATCH_4BITS(Quantized4Asymm, Quantized4Asymm);
#undef DISPATCH_4BITS
#undef DISPATCH_RAW
    megdnn_assert(false,
                  "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                  stype.name(), dtype.name(), h, w);
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
    after_kernel_launch();
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
    after_kernel_launch();
}
