/**
 * \file dnn/src/cuda/relayout_format/relayout_format_nchw_nhwc.cu
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
template <int pack_w>
struct rwtype_helper;

template <>
struct rwtype_helper<2> {
    using InnerDtype = char;
};

template <>
struct rwtype_helper<8> {
    using InnerDtype = unsigned;
};
}  // namespace

void relayout_format::relayout_format_cuda_nchw_nhwc(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point) {
    auto&& stype = src.layout.dtype;
    auto&& dtype = dst.layout.dtype;
    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;
    int n = src.layout[0];
    int ic = src.layout[1];
    int h = src.layout[2];
    int w = src.layout[3];
    int w_pad = DIVUP(w, 2) * 2;
    int hw = h * w_pad;
    int n_stride_src = src_layout.stride[0];
    int ic_stride = src_layout.stride[1];
    int n_stride_dst = dst_layout.stride[0];
    int hw_stride = dst_layout.stride[2];
    static constexpr int chan_blk = 8;
    static constexpr int pack_oc = 8;
    int problem_size = n * DIVUP(ic, chan_blk) * hw;
    int oc = dst.layout[3];

    bool same_scale = src_scale == dst_scale;
    bool padding = w % 2 != 0;
#define DISPATCH_RAW(_padding, _same_scale, _pack_w, _src_type, _dst_type,     \
                     _src_c_type, _dst_c_type, _size_nbits)                    \
    if (padding == _padding && same_scale == _same_scale &&                    \
        hw % _pack_w == 0 && stype.enumv().ev == DTypeEnum::Ev::_src_type &&   \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                        \
        using InnerDtype_ = typename rwtype_helper<_pack_w>::InnerDtype;       \
        using SrcIterator_ =                                                   \
                TensorIteratorOverChannel<InnerDtype_, 1, chan_blk, _pack_w,   \
                                          _size_nbits>;                        \
        using DstIterator_ = typename TensorIteratorPolicy<                    \
                _padding, _dst_c_type, pack_oc, chan_blk, _pack_w,             \
                _size_nbits, LayoutType::NHWC>::TensorIterator;                \
        using CudaPostProcess_ =                                               \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,            \
                                _same_scale>;                                  \
        using Transpose_ =                                                     \
                Translayout<_pack_w, chan_blk, InnerDtype_, dtype::_src_type,  \
                            dtype::_dst_type, _same_scale>;                    \
        using RelayoutProblem_ =                                               \
                RelayoutProblem<SrcIterator_, DstIterator_, Transpose_,        \
                                CudaPostProcess_>;                             \
        n_stride_src = n_stride_src * _size_nbits / (8 * sizeof(InnerDtype_)); \
        ic_stride = ic_stride * _size_nbits / (8 * sizeof(InnerDtype_));       \
        n_stride_dst = n_stride_dst * _size_nbits / (8 * sizeof(_dst_c_type)); \
        hw_stride = hw_stride * _size_nbits / (8 * sizeof(_dst_c_type));       \
        typename RelayoutProblem_::Param param{                                \
                SrcIterator_{(InnerDtype_*)src.raw_ptr, ic_stride, ic, w,      \
                             w_pad},                                           \
                DstIterator_{(_dst_c_type*)dst.raw_ptr, hw_stride, oc, w,      \
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
#define DISPATCH_4BITS(_src_type, _dst_type)                            \
    DISPATCH_RAW(true, true, 8, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 8, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(true, true, 2, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 2, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, true, 8, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 8, _src_type, _dst_type, char, char, 4); \
    DISPATCH_RAW(false, true, 2, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 2, _src_type, _dst_type, char, char, 4);
    DISPATCH_4BITS(QuantizedS4, QuantizedS4);
    DISPATCH_4BITS(Quantized4Asymm, Quantized4Asymm);
#undef DISPATCH_4BITS
#undef DISPATCH_RAW
    megdnn_assert(false,
                  "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                  stype.name(), dtype.name(), h, w);
}

void relayout_format::relayout_format_cuda_nhwc_nchw(
        const TensorND& src, const TensorND& dst, const cudaStream_t& stream,
        const float src_scale, const float dst_scale,
        const uint8_t src_zero_point, const uint8_t dst_zero_point) {
    auto&& stype = src.layout.dtype;
    auto&& dtype = dst.layout.dtype;
    auto& src_layout = src.layout;
    auto& dst_layout = dst.layout;

    int n = src.layout[0];
    int h = src.layout[1];
    int w = src.layout[2];
    int ic = src.layout[3];
    int w_pad = DIVUP(w, 2) * 2;
    int hw = h * w_pad;
    int n_stride_src = src_layout.stride[0];
    int hw_stride = src_layout.stride[2];
    int n_stride_dst = dst_layout.stride[0];
    int oc_stride = dst_layout.stride[1];
    static constexpr int chan_blk = 8;
    static constexpr int pack_oc = 8;
    int problem_size = n * DIVUP(ic, chan_blk) * hw;
    int oc = dst.layout[1];

    bool same_scale = src_scale == dst_scale;
    bool padding = w % 2 != 0;
#define DISPATCH_RAW(_padding, _same_scale, _pack_w, _src_type, _dst_type,     \
                     _src_c_type, _dst_c_type, _size_nbits)                    \
    if (padding == _padding && same_scale == _same_scale &&                    \
        hw % _pack_w == 0 && stype.enumv().ev == DTypeEnum::Ev::_src_type &&   \
        dtype.enumv().ev == DTypeEnum::Ev::_dst_type) {                        \
        using SrcIterator_ = typename TensorIteratorPolicy<                    \
                _padding, _src_c_type, pack_oc, chan_blk, _pack_w,             \
                _size_nbits, LayoutType::NHWC>::TensorIterator;                \
        using InnerDtype_ = typename rwtype_helper<_pack_w>::InnerDtype;       \
        using DstIterator_ =                                                   \
                TensorIteratorOverChannel<InnerDtype_, 1, chan_blk, _pack_w,   \
                                          _size_nbits>;                        \
        using CudaPostProcess_ =                                               \
                CudaPostProcess<dtype::_src_type, dtype::_dst_type,            \
                                _same_scale>;                                  \
        using Transpose_ =                                                     \
                Translayout<chan_blk, _pack_w, _src_c_type, dtype::_src_type,  \
                            dtype::_dst_type, _same_scale>;                    \
        using RelayoutProblem_ =                                               \
                RelayoutProblem<SrcIterator_, DstIterator_, Transpose_,        \
                                CudaPostProcess_>;                             \
        n_stride_src = n_stride_src * _size_nbits / (8 * sizeof(_src_c_type)); \
        hw_stride = hw_stride * _size_nbits / (8 * sizeof(_src_c_type));       \
        n_stride_dst = n_stride_dst * _size_nbits / (8 * sizeof(InnerDtype_)); \
        oc_stride = oc_stride * _size_nbits / (8 * sizeof(InnerDtype_));       \
        typename RelayoutProblem_::Param param{                                \
                SrcIterator_{(_src_c_type*)src.raw_ptr, hw_stride, ic, w,      \
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
#define DISPATCH_4BITS(_src_type, _dst_type)                            \
    DISPATCH_RAW(true, true, 8, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 8, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(true, true, 2, _src_type, _dst_type, char, char, 4);   \
    DISPATCH_RAW(true, false, 2, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, true, 8, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 8, _src_type, _dst_type, char, char, 4); \
    DISPATCH_RAW(false, true, 2, _src_type, _dst_type, char, char, 4);  \
    DISPATCH_RAW(false, false, 2, _src_type, _dst_type, char, char, 4);
    DISPATCH_4BITS(QuantizedS4, QuantizedS4);
    DISPATCH_4BITS(Quantized4Asymm, Quantized4Asymm);
#undef DISPATCH_4BITS
#undef DISPATCH_RAW
    megdnn_assert(false,
                  "Unsupported data type(src:%s, dst:%s) or image size(%dx%d).",
                  stype.name(), dtype.name(), h, w);
}
