/**
 * \file dnn/src/cuda/relayout_format/translayout.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/cuda/integer_subbyte_utils.cuh"
#include "src/cuda/relayout_format/cuda_post_process.cuh"
#include "src/cuda/relayout_format/relayout_format.cuh"
#include "src/cuda/relayout_format/relayout_format_utils.cuh"

namespace megdnn {
namespace cuda {
namespace relayout_format {
namespace internal {
using namespace integer_subbyte;

template <typename dt>
struct qtype_signedness;

template <>
struct qtype_signedness<dtype::QuantizedS4> {
    static constexpr bool value = true;
};

template <>
struct qtype_signedness<dtype::Quantized4Asymm> {
    static constexpr bool value = false;
};

template <typename dt_src, typename dt_dst>
struct enable_qtype_b4 {
    static constexpr bool val_src =
            std::is_same<dt_src, dtype::QuantizedS4>::value ||
            std::is_same<dt_src, dtype::Quantized4Asymm>::value;
    static constexpr bool val_dst =
            std::is_same<dt_dst, dtype::QuantizedS4>::value ||
            std::is_same<dt_dst, dtype::Quantized4Asymm>::value;
    using type = typename std::enable_if<std::is_same<dt_src, dt_dst>::value &&
                                         val_src && val_dst>::type;
};

// The input fragment is stored in RowMajor order. The translayout operator
// performs a transpose operation on the input fragment, and produces a
// reordered fragment, i.e. a fragment stored in ColumnMajor order.
template <int col, int row, typename SrcType, typename DnnSrcType,
          typename DnnDstType, bool same_scale, typename enable = void>
struct Translayout;

// partial specialization for translayout operator for qint8 and quint8
template <typename SrcType, typename DnnSrcType, typename DnnDstType,
          bool same_scale>
struct Translayout<1, 4, SrcType, DnnSrcType, DnnDstType, same_scale> {
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   1>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   1>::DstDtype;
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
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   4>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   4>::DstDtype;
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

// =========================================================

// partial specialization for translayout operator for qint4
// NCHW <-> NCHW64
template <typename SrcType, typename DnnSrcType_, typename DnnDstType_,
          bool same_scale>
struct Translayout<2, 64, SrcType, DnnSrcType_, DnnDstType_, same_scale,
                   typename enable_qtype_b4<DnnSrcType_, DnnDstType_>::type> {
    using DnnSrcType = DnnSrcType_;
    using DnnDstType = DnnDstType_;
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   2>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   2>::DstDtype;
    static constexpr bool signedness = qtype_signedness<DnnSrcType>::value;
    static inline __device__ void trans(
            DstDtype (&dst_width)[2], InnerDtype (&read_channel)[64],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][2];
        int* dst_frag = reinterpret_cast<int*>(dst_width);
        auto pack_channel = [&](int idx) -> int {
            return transform_int8_to_b4x8<signedness>(
                    post_process(intermediate[0][idx]),
                    post_process(intermediate[1][idx]),
                    post_process(intermediate[2][idx]),
                    post_process(intermediate[3][idx]),
                    post_process(intermediate[4][idx]),
                    post_process(intermediate[5][idx]),
                    post_process(intermediate[6][idx]),
                    post_process(intermediate[7][idx]));
        };
#pragma unroll
        for (int i = 0; i < 64; i += 8) {
            transform_b4x2_to_int8<signedness>(
                    intermediate[0],
                    reinterpret_cast<uint8_t&>(read_channel[i + 0]));
            transform_b4x2_to_int8<signedness>(
                    intermediate[1],
                    reinterpret_cast<uint8_t&>(read_channel[i + 1]));
            transform_b4x2_to_int8<signedness>(
                    intermediate[2],
                    reinterpret_cast<uint8_t&>(read_channel[i + 2]));
            transform_b4x2_to_int8<signedness>(
                    intermediate[3],
                    reinterpret_cast<uint8_t&>(read_channel[i + 3]));
            transform_b4x2_to_int8<signedness>(
                    intermediate[4],
                    reinterpret_cast<uint8_t&>(read_channel[i + 4]));
            transform_b4x2_to_int8<signedness>(
                    intermediate[5],
                    reinterpret_cast<uint8_t&>(read_channel[i + 5]));
            transform_b4x2_to_int8<signedness>(
                    intermediate[6],
                    reinterpret_cast<uint8_t&>(read_channel[i + 6]));
            transform_b4x2_to_int8<signedness>(
                    intermediate[7],
                    reinterpret_cast<uint8_t&>(read_channel[i + 7]));

            int frag_idx = i / 8;
            dst_frag[0 * 8 + frag_idx] = pack_channel(0);
            dst_frag[1 * 8 + frag_idx] = pack_channel(1);
        }
    }
    using Fragment = array_wrapper<SrcType, 64>;
    static inline __device__ void trans(
            Fragment& dst, Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        trans(reinterpret_cast<DstDtype(&)[2]>(dst),
              reinterpret_cast<InnerDtype(&)[64]>(src), post_process, 0);
    }
};

template <typename SrcType, typename DnnSrcType_, typename DnnDstType_,
          bool same_scale>
struct Translayout<8, 64, SrcType, DnnSrcType_, DnnDstType_, same_scale,
                   typename enable_qtype_b4<DnnSrcType_, DnnDstType_>::type> {
    using DnnSrcType = DnnSrcType_;
    using DnnDstType = DnnDstType_;
    using InnerDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   8>::InnerDtype;
    using DstDtype =
            typename DTypeRWHelper<typename DTypeTrait<DnnSrcType>::ctype,
                                   8>::DstDtype;
    static constexpr bool signedness = qtype_signedness<DnnSrcType>::value;
    static inline __device__ void trans(
            DstDtype (&dst_width)[8], InnerDtype (&read_channel)[64],
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][8];
        int* dst_frag = reinterpret_cast<int*>(dst_width);
        auto pack_channel = [&](int idx) -> int {
            return transform_int8_to_b4x8<signedness>(
                    post_process(intermediate[0][idx]),
                    post_process(intermediate[1][idx]),
                    post_process(intermediate[2][idx]),
                    post_process(intermediate[3][idx]),
                    post_process(intermediate[4][idx]),
                    post_process(intermediate[5][idx]),
                    post_process(intermediate[6][idx]),
                    post_process(intermediate[7][idx]));
        };
#pragma unroll
        for (int i = 0; i < 64; i += 8) {
            transform_b4x8_to_int8<signedness>(intermediate[0],
                                               read_channel[i + 0]);
            transform_b4x8_to_int8<signedness>(intermediate[1],
                                               read_channel[i + 1]);
            transform_b4x8_to_int8<signedness>(intermediate[2],
                                               read_channel[i + 2]);
            transform_b4x8_to_int8<signedness>(intermediate[3],
                                               read_channel[i + 3]);
            transform_b4x8_to_int8<signedness>(intermediate[4],
                                               read_channel[i + 4]);
            transform_b4x8_to_int8<signedness>(intermediate[5],
                                               read_channel[i + 5]);
            transform_b4x8_to_int8<signedness>(intermediate[6],
                                               read_channel[i + 6]);
            transform_b4x8_to_int8<signedness>(intermediate[7],
                                               read_channel[i + 7]);
            int frag_idx = i / 8;
            dst_frag[0 * 8 + frag_idx] = pack_channel(0);
            dst_frag[1 * 8 + frag_idx] = pack_channel(1);
            dst_frag[2 * 8 + frag_idx] = pack_channel(2);
            dst_frag[3 * 8 + frag_idx] = pack_channel(3);
            dst_frag[4 * 8 + frag_idx] = pack_channel(4);
            dst_frag[5 * 8 + frag_idx] = pack_channel(5);
            dst_frag[6 * 8 + frag_idx] = pack_channel(6);
            dst_frag[7 * 8 + frag_idx] = pack_channel(7);
        }
    }
    using Fragment = array_wrapper<unsigned, 64>;
    static inline __device__ void trans(
            Fragment& dst, Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        trans(reinterpret_cast<DstDtype(&)[8]>(dst),
              reinterpret_cast<InnerDtype(&)[64]>(src), post_process, 0);
    }
};

template <typename SrcType, typename DnnSrcType_, typename DnnDstType_,
          bool same_scale>
struct Translayout<64, 8, SrcType, DnnSrcType_, DnnDstType_, same_scale,
                   typename enable_qtype_b4<DnnSrcType_, DnnDstType_>::type> {
    using DnnSrcType = DnnSrcType_;
    using DnnDstType = DnnDstType_;
    static constexpr int row = 8;
    static constexpr int col = 64;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr int inc_col = 8;
    static constexpr int inc_col_in_type =
            inc_col * size_nbits / (8 * sizeof(SrcType));
    static constexpr bool signedness = qtype_signedness<DnnSrcType>::value;
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static MEGDNN_DEVICE __forceinline__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        int intermediate[8][8];
        int* dst_frag = reinterpret_cast<int*>(&dst);
        auto pack = [&](int idx) -> int {
            return transform_int8_to_b4x8<signedness>(
                    post_process(intermediate[0][idx]),
                    post_process(intermediate[1][idx]),
                    post_process(intermediate[2][idx]),
                    post_process(intermediate[3][idx]),
                    post_process(intermediate[4][idx]),
                    post_process(intermediate[5][idx]),
                    post_process(intermediate[6][idx]),
                    post_process(intermediate[7][idx]));
        };
#pragma unroll
        for (int j = 0; j < col_in_type; j += inc_col_in_type) {
            transform_b4x8_to_int8<signedness>(
                    intermediate[0],
                    reinterpret_cast<const int&>(src[0 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[1],
                    reinterpret_cast<const int&>(src[1 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[2],
                    reinterpret_cast<const int&>(src[2 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[3],
                    reinterpret_cast<const int&>(src[3 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[4],
                    reinterpret_cast<const int&>(src[4 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[5],
                    reinterpret_cast<const int&>(src[5 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[6],
                    reinterpret_cast<const int&>(src[6 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[7],
                    reinterpret_cast<const int&>(src[7 * col_in_type + j]));
            dst_frag[(j / inc_col_in_type) * 8 + 0] = pack(0);
            dst_frag[(j / inc_col_in_type) * 8 + 1] = pack(1);
            dst_frag[(j / inc_col_in_type) * 8 + 2] = pack(2);
            dst_frag[(j / inc_col_in_type) * 8 + 3] = pack(3);
            dst_frag[(j / inc_col_in_type) * 8 + 4] = pack(4);
            dst_frag[(j / inc_col_in_type) * 8 + 5] = pack(5);
            dst_frag[(j / inc_col_in_type) * 8 + 6] = pack(6);
            dst_frag[(j / inc_col_in_type) * 8 + 7] = pack(7);
        }
    }
};

template <typename SrcType, typename DnnSrcType_, typename DnnDstType_,
          bool same_scale>
struct Translayout<64, 2, SrcType, DnnSrcType_, DnnDstType_, same_scale,
                   typename enable_qtype_b4<DnnSrcType_, DnnDstType_>::type> {
    using DnnSrcType = DnnSrcType_;
    using DnnDstType = DnnDstType_;
    static constexpr int row = 2;
    static constexpr int col = 64;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr int inc_col = 8;
    static constexpr int inc_col_in_type =
            inc_col * size_nbits / (8 * sizeof(SrcType));
    static constexpr bool signedness = qtype_signedness<DnnSrcType>::value;
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static MEGDNN_DEVICE __forceinline__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process) {
        int intermediate[2][8];
        int* dst_frag = reinterpret_cast<int*>(&dst);
#pragma unroll
        for (int j = 0; j < col_in_type; j += inc_col_in_type) {
            transform_b4x8_to_int8<signedness>(
                    intermediate[0],
                    reinterpret_cast<const int&>(src[0 * col_in_type + j]));
            transform_b4x8_to_int8<signedness>(
                    intermediate[1],
                    reinterpret_cast<const int&>(src[1 * col_in_type + j]));
            dst_frag[(j / inc_col_in_type) * 2 + 0] =
                    transform_int8_to_b4x8<signedness>(
                            post_process(intermediate[0][0]),
                            post_process(intermediate[1][0]),
                            post_process(intermediate[0][1]),
                            post_process(intermediate[1][1]),
                            post_process(intermediate[0][2]),
                            post_process(intermediate[1][2]),
                            post_process(intermediate[0][3]),
                            post_process(intermediate[1][3]));
            dst_frag[(j / inc_col_in_type) * 2 + 1] =
                    transform_int8_to_b4x8<signedness>(
                            post_process(intermediate[0][4]),
                            post_process(intermediate[1][4]),
                            post_process(intermediate[0][5]),
                            post_process(intermediate[1][5]),
                            post_process(intermediate[0][6]),
                            post_process(intermediate[1][6]),
                            post_process(intermediate[0][7]),
                            post_process(intermediate[1][7]));
        }
    }
};
// =========================================================

// partial specialization for translayout operator for qint4
// NCHW <-> NHWC
template <typename SrcType, typename DnnSrcType_, typename DnnDstType_,
          bool same_scale>
struct Translayout<2, 8, SrcType, DnnSrcType_, DnnDstType_, same_scale,
                   typename enable_qtype_b4<DnnSrcType_, DnnDstType_>::type> {
    using DnnSrcType = DnnSrcType_;
    using DnnDstType = DnnDstType_;
    static constexpr int row = 8;
    static constexpr int col = 2;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr bool signedness = qtype_signedness<DnnSrcType>::value;
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static inline __device__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][2];
        transform_b4x2_to_int8<signedness>(intermediate[0],
                                           reinterpret_cast<uint8_t&>(src[0]));
        transform_b4x2_to_int8<signedness>(intermediate[1],
                                           reinterpret_cast<uint8_t&>(src[1]));
        transform_b4x2_to_int8<signedness>(intermediate[2],
                                           reinterpret_cast<uint8_t&>(src[2]));
        transform_b4x2_to_int8<signedness>(intermediate[3],
                                           reinterpret_cast<uint8_t&>(src[3]));
        transform_b4x2_to_int8<signedness>(intermediate[4],
                                           reinterpret_cast<uint8_t&>(src[4]));
        transform_b4x2_to_int8<signedness>(intermediate[5],
                                           reinterpret_cast<uint8_t&>(src[5]));
        transform_b4x2_to_int8<signedness>(intermediate[6],
                                           reinterpret_cast<uint8_t&>(src[6]));
        transform_b4x2_to_int8<signedness>(intermediate[7],
                                           reinterpret_cast<uint8_t&>(src[7]));

        int* dst_frag = reinterpret_cast<int*>(&dst);
        auto pack = [&](int idx) -> int {
            return transform_int8_to_b4x8<signedness>(
                    post_process(intermediate[0][idx]),
                    post_process(intermediate[1][idx]),
                    post_process(intermediate[2][idx]),
                    post_process(intermediate[3][idx]),
                    post_process(intermediate[4][idx]),
                    post_process(intermediate[5][idx]),
                    post_process(intermediate[6][idx]),
                    post_process(intermediate[7][idx]));
        };
        dst_frag[0] = pack(0);
        dst_frag[1] = pack(1);
    }
};

template <typename SrcType, typename DnnSrcType_, typename DnnDstType_,
          bool same_scale>
struct Translayout<8, 8, SrcType, DnnSrcType_, DnnDstType_, same_scale,
                   typename enable_qtype_b4<DnnSrcType_, DnnDstType_>::type> {
    using DnnSrcType = DnnSrcType_;
    using DnnDstType = DnnDstType_;
    static constexpr int row = 8;
    static constexpr int col = 8;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr bool signedness = qtype_signedness<DnnSrcType>::value;
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static inline __device__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[8][8];
        transform_b4x8_to_int8<signedness>(
                intermediate[0], reinterpret_cast<const int&>(src[0]));
        transform_b4x8_to_int8<signedness>(
                intermediate[1], reinterpret_cast<const int&>(src[1]));
        transform_b4x8_to_int8<signedness>(
                intermediate[2], reinterpret_cast<const int&>(src[2]));
        transform_b4x8_to_int8<signedness>(
                intermediate[3], reinterpret_cast<const int&>(src[3]));
        transform_b4x8_to_int8<signedness>(
                intermediate[4], reinterpret_cast<const int&>(src[4]));
        transform_b4x8_to_int8<signedness>(
                intermediate[5], reinterpret_cast<const int&>(src[5]));
        transform_b4x8_to_int8<signedness>(
                intermediate[6], reinterpret_cast<const int&>(src[6]));
        transform_b4x8_to_int8<signedness>(
                intermediate[7], reinterpret_cast<const int&>(src[7]));
        int* dst_frag = reinterpret_cast<int*>(&dst);
        auto pack = [&](int idx) {
            return transform_int8_to_b4x8<signedness>(
                    post_process(intermediate[0][idx]),
                    post_process(intermediate[1][idx]),
                    post_process(intermediate[2][idx]),
                    post_process(intermediate[3][idx]),
                    post_process(intermediate[4][idx]),
                    post_process(intermediate[5][idx]),
                    post_process(intermediate[6][idx]),
                    post_process(intermediate[7][idx]));
        };
        dst_frag[0] = pack(0);
        dst_frag[1] = pack(1);
        dst_frag[2] = pack(2);
        dst_frag[3] = pack(3);
        dst_frag[4] = pack(4);
        dst_frag[5] = pack(5);
        dst_frag[6] = pack(6);
        dst_frag[7] = pack(7);
    }
};

template <typename SrcType, typename DnnSrcType_, typename DnnDstType_,
          bool same_scale>
struct Translayout<8, 2, SrcType, DnnSrcType_, DnnDstType_, same_scale,
                   typename enable_qtype_b4<DnnSrcType_, DnnDstType_>::type> {
    using DnnSrcType = DnnSrcType_;
    using DnnDstType = DnnDstType_;
    static constexpr int row = 2;
    static constexpr int col = 8;
    static constexpr int size_nbits = 4;
    static constexpr int col_in_type = col * size_nbits / (8 * sizeof(SrcType));
    static constexpr int elements_in_type = row * col_in_type;
    static constexpr bool signedness = qtype_signedness<DnnSrcType>::value;
    using Fragment = array_wrapper<SrcType, elements_in_type>;
    static inline __device__ void trans(
            Fragment& dst, const Fragment& src,
            CudaPostProcess<DnnSrcType, DnnDstType, same_scale>& post_process,
            const char zero_point) {
        int intermediate[2][8];
        transform_b4x8_to_int8<signedness>(
                intermediate[0], reinterpret_cast<const int&>(src[0]));
        transform_b4x8_to_int8<signedness>(
                intermediate[1], reinterpret_cast<const int&>(src[1]));
        int* dst_frag = reinterpret_cast<int*>(&dst);
        dst_frag[0] = transform_int8_to_b4x8<signedness>(
                post_process(intermediate[0][0]),
                post_process(intermediate[1][0]),
                post_process(intermediate[0][1]),
                post_process(intermediate[1][1]),
                post_process(intermediate[0][2]),
                post_process(intermediate[1][2]),
                post_process(intermediate[0][3]),
                post_process(intermediate[1][3]));
        dst_frag[1] = transform_int8_to_b4x8<signedness>(
                post_process(intermediate[0][4]),
                post_process(intermediate[1][4]),
                post_process(intermediate[0][5]),
                post_process(intermediate[1][5]),
                post_process(intermediate[0][6]),
                post_process(intermediate[1][6]),
                post_process(intermediate[0][7]),
                post_process(intermediate[1][7]));
    }
};

}  // namespace internal
}  // namespace relayout_format
}  // namespace cuda
}  // namespace megdnn
