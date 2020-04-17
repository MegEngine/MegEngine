/**
 * \file dnn/src/x86/elemwise_op.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "src/x86/elemwise_helper/op_binary.h"
#include "src/x86/elemwise_helper/op_ternary.h"
#include "src/x86/elemwise_helper/op_unary.h"
#include "src/x86/utils.h"

namespace megdnn {
namespace x86 {

///////////////////////////////// ParamElemVistor ///////////////////////////
template <typename ctype, SIMDType simd_type = SIMDType::SSE4_2>
struct ParamElemVisitor;

//! visitor single elemwise, and dup to vector
template <typename ctype, SIMDType simd_type = SIMDType::SSE4_2>
struct ParamElemVisitorDup;

#define cb(_ctype, _simd_ptr_type, _simd_target, _src_ptr_ctype, _simd_type, \
           _fun_prefix, _fun_suffix1, _fun_suffix2, simd_type)               \
    template <>                                                              \
    struct ParamElemVisitor<_ctype, simd_type> {                             \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                \
        _simd_type operator()(const _ctype* src) const {                     \
            return _##_fun_prefix##_loadu_##_fun_suffix1(                    \
                    reinterpret_cast<const _simd_ptr_type*>(src));           \
        }                                                                    \
    };                                                                       \
    template <>                                                              \
    struct ParamElemVisitorDup<_ctype, simd_type> {                          \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                \
        _simd_type operator()(const _ctype* src) const {                     \
            return _##_fun_prefix##_set1_##_fun_suffix2(                     \
                    *reinterpret_cast<const _src_ptr_ctype*>(src));          \
        }                                                                    \
    }

cb(dt_qint32, __m128i, "sse4.2", int, __m128i, mm, si128, epi32,
   SIMDType::SSE4_2);
cb(dt_qint8, __m128i, "sse4.2", int8_t, __m128i, mm, si128, epi8,
   SIMDType::SSE4_2);
cb(dt_quint8, __m128i, "sse4.2", uint8_t, __m128i, mm, si128, epi8,
   SIMDType::SSE4_2);
cb(dt_int32, __m128i, "sse4.2", int32_t, __m128i, mm, si128, epi32,
   SIMDType::SSE4_2);
cb(dt_int16, __m128i, "sse4.2", short, __m128i, mm, si128, epi16,
   SIMDType::SSE4_2);
cb(dt_int8, __m128i, "sse4.2", int8_t, __m128i, mm, si128, epi8,
   SIMDType::SSE4_2);
cb(dt_uint8, __m128i, "sse4.2", uint8_t, __m128i, mm, si128, epi8,
   SIMDType::SSE4_2);
cb(dt_float32, float, "sse4.2", float, __m128, mm, ps, ps, SIMDType::SSE4_2);

cb(dt_qint32, __m256i, "avx2", int, __m256i, mm256, si256, epi32,
   SIMDType::AVX2);
cb(dt_qint8, __m256i, "avx2", int8_t, __m256i, mm256, si256, epi8,
   SIMDType::AVX2);
cb(dt_quint8, __m256i, "avx2", uint8_t, __m256i, mm256, si256, epi8,
   SIMDType::AVX2);
cb(dt_int32, __m256i, "avx2", int, __m256i, mm256, si256, epi32,
   SIMDType::AVX2);
cb(dt_int16, __m256i, "avx2", short, __m256i, mm256, si256, epi16,
   SIMDType::AVX2);
cb(dt_int8, __m256i, "avx2", int8_t, __m256i, mm256, si256, epi8,
   SIMDType::AVX2);
cb(dt_uint8, __m256i, "avx2", uint8_t, __m256i, mm256, si256, epi8,
   SIMDType::AVX2);
cb(dt_float32, float, "avx2", float, __m256, mm256, ps, ps, SIMDType::AVX2);

#undef cb
/*!
 * \brief broadcast type
 * BCAST_x[0]x[1]...: x[i] == !stride[i]
 */
enum BcastType {
    VEC,
    VEC_VEC,
    VEC_BCAST101,
    VEC_SCALAR,
    SCALAR_VEC,
    BCAST101_VEC,
    BCAST101x_VEC,  // used for nchwxx bias add, 1c18
    VEC_VEC_VEC,
    VEC_VEC_SCALAR,
    BCAST101_VEC_BCAST101,
    VEC_BCAST101_VEC,
    VEC_SCALAR_VEC,
    VEC_SCALAR_SCALAR
};

///////////////////////////////// OpCaller /////////////////////////////
template <typename Op, SIMDType simd_type>
struct OpCallerUnary;

#define OP_CALLER(simd_type, target_simd)                             \
    template <typename Op>                                            \
    struct OpCallerUnary<Op, simd_type> {                             \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                          \
        static void run(const typename Op::src_ctype* src,            \
                        typename Op::dst_ctype* dst, DType src_dtype, \
                        DType dst_dtype, size_t nr_elems) {           \
            Op op(src_dtype, dst_dtype);                              \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis;  \
            size_t i = 0;                                             \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                \
                 i += Op::SIMD_WIDTH * 2) {                           \
                op({{vis(src), vis(src + Op::SIMD_WIDTH)}}, dst);     \
                src += Op::SIMD_WIDTH * 2;                            \
                dst += Op::SIMD_WIDTH * 2;                            \
            }                                                         \
            for (; i < nr_elems; i++) {                               \
                op(*src, dst);                                        \
                src++;                                                \
                dst++;                                                \
            }                                                         \
        }                                                             \
    };

OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")
template <typename Op>
struct OpCallerUnary<Op, SIMDType::NONE> {
    static void run(const typename Op::src_ctype* src,
                    typename Op::dst_ctype* dst, DType src_dtype,
                    DType dst_dtype, size_t nr_elems) {
        Op op(src_dtype, dst_dtype);
        for (size_t i = 0; i < nr_elems; i++) {
            op(*src, dst);
            src++;
            dst++;
        }
    }
};
#undef OP_CALLER

template <typename Op, SIMDType simd_type, BcastType bcast_type>
struct OpCallerBinary;

#define OP_CALLER(simd_type, target_simd)                                     \
    template <typename Op>                                                    \
    struct OpCallerBinary<Op, simd_type, VEC_VEC> {                           \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                  \
        static void run(const typename Op::src_ctype* src0,                   \
                        const typename Op::src_ctype* src1,                   \
                        typename Op::dst_ctype* dst, DType src0_dtype,        \
                        DType src1_dtype, DType dst_dtype, size_t nr_elems) { \
            Op op(src0_dtype, src1_dtype, dst_dtype);                         \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;         \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis1;         \
            size_t i = 0;                                                     \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                        \
                 i += Op::SIMD_WIDTH * 2) {                                   \
                op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},               \
                   {{vis1(src1), vis1(src1 + Op::SIMD_WIDTH)}}, dst);         \
                src0 += Op::SIMD_WIDTH * 2;                                   \
                src1 += Op::SIMD_WIDTH * 2;                                   \
                dst += Op::SIMD_WIDTH * 2;                                    \
            }                                                                 \
            for (; i < nr_elems; i++) {                                       \
                op(*src0, *src1, dst);                                        \
                src0++;                                                       \
                src1++;                                                       \
                dst++;                                                        \
            }                                                                 \
        }                                                                     \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")
template <typename Op>
struct OpCallerBinary<Op, SIMDType::NONE, VEC_VEC> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType dst_dtype, size_t nr_elems) {
        Op op(src0_dtype, src1_dtype, dst_dtype);
        size_t i = 0;
        for (; i < nr_elems; i++) {
            op(*src0, *src1, dst);
            src0++;
            src1++;
            dst++;
        }
    }
};
#undef OP_CALLER

#define OP_CALLER(simd_type, target_simd)                                \
    template <typename Op>                                               \
    struct OpCallerBinary<Op, simd_type, VEC_BCAST101> {                 \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                             \
        static void run(const typename Op::src_ctype* src0,              \
                        const typename Op::src_ctype* src1,              \
                        typename Op::dst_ctype* dst, DType src0_dtype,   \
                        DType src1_dtype, DType dst_dtype, size_t batch, \
                        size_t channel, size_t channel_stride) {         \
            Op op(src0_dtype, src1_dtype, dst_dtype);                    \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;    \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis1; \
            for (size_t b = 0; b < batch; b++) {                         \
                const typename Op::src_ctype* src1_ptr = src1;           \
                for (size_t c = 0; c < channel; c++) {                   \
                    size_t i = 0;                                        \
                    auto src1_simd = vis1(src1_ptr);                     \
                    for (; i + Op::SIMD_WIDTH * 2 <= channel_stride;     \
                         i += Op::SIMD_WIDTH * 2) {                      \
                        op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},  \
                           {{src1_simd, src1_simd}}, dst);               \
                        src0 += Op::SIMD_WIDTH * 2;                      \
                        dst += Op::SIMD_WIDTH * 2;                       \
                    }                                                    \
                    for (; i < channel_stride; i++) {                    \
                        op(*src0, *src1_ptr, dst);                       \
                        src0++;                                          \
                        dst++;                                           \
                    }                                                    \
                    src1_ptr++;                                          \
                }                                                        \
            }                                                            \
        }                                                                \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")
template <typename Op>
struct OpCallerBinary<Op, SIMDType::NONE, VEC_BCAST101> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType dst_dtype, size_t batch,
                    size_t channel, size_t channel_stride) {
        Op op(src0_dtype, src1_dtype, dst_dtype);
        for (size_t b = 0; b < batch; b++) {
            const typename Op::src_ctype* src1_ptr = src1;
            for (size_t c = 0; c < channel; c++) {
                for (size_t i = 0; i < channel_stride; i++) {
                    op(*src0, *src1_ptr, dst);
                    src0++;
                    dst++;
                }
                src1_ptr++;
            }
        }
    }
};
#undef OP_CALLER

#define OP_CALLER(simd_type, target_simd)                                     \
    template <typename Op>                                                    \
    struct OpCallerBinary<Op, simd_type, VEC_SCALAR> {                        \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                  \
        static void run(const typename Op::src_ctype* src0,                   \
                        const typename Op::src_ctype src1,                    \
                        typename Op::dst_ctype* dst, DType src0_dtype,        \
                        DType src1_dtype, DType dst_dtype, size_t nr_elems) { \
            Op op(src0_dtype, src1_dtype, dst_dtype);                         \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;         \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis1;      \
            auto vis1_simd = vis1(&src1);                                     \
            size_t i = 0;                                                     \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                        \
                 i += Op::SIMD_WIDTH * 2) {                                   \
                op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},               \
                   {{vis1_simd, vis1_simd}}, dst);                            \
                src0 += Op::SIMD_WIDTH * 2;                                   \
                dst += Op::SIMD_WIDTH * 2;                                    \
            }                                                                 \
            for (; i < nr_elems; i++) {                                       \
                op(*src0, src1, dst);                                         \
                src0++;                                                       \
                dst++;                                                        \
            }                                                                 \
        }                                                                     \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")
template <typename Op>
struct OpCallerBinary<Op, SIMDType::NONE, VEC_SCALAR> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype src1,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType dst_dtype, size_t nr_elems) {
        Op op(src0_dtype, src1_dtype, dst_dtype);
        size_t i = 0;
        for (; i < nr_elems; i++) {
            op(*src0, src1, dst);
            src0++;
            dst++;
        }
    }
};
#undef OP_CALLER

//! this only for nonswap op, like SUB and DIV
#define OP_CALLER(simd_type, target_simd)                                     \
    template <typename Op>                                                    \
    struct OpCallerBinary<Op, simd_type, SCALAR_VEC> {                        \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                  \
        static void run(const typename Op::src_ctype src0,                    \
                        const typename Op::src_ctype* src1,                   \
                        typename Op::dst_ctype* dst, DType src0_dtype,        \
                        DType src1_dtype, DType dst_dtype, size_t nr_elems) { \
            Op op(src0_dtype, src1_dtype, dst_dtype);                         \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis0;      \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis1;         \
            auto vis0_simd = vis0(&src0);                                     \
            size_t i = 0;                                                     \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                        \
                 i += Op::SIMD_WIDTH * 2) {                                   \
                op({{vis0_simd, vis0_simd}},                                  \
                   {{vis1(src1), vis1(src1 + Op::SIMD_WIDTH)}}, dst);         \
                src1 += Op::SIMD_WIDTH * 2;                                   \
                dst += Op::SIMD_WIDTH * 2;                                    \
            }                                                                 \
            for (; i < nr_elems; i++) {                                       \
                op(src0, *src1, dst);                                         \
                src1++;                                                       \
                dst++;                                                        \
            }                                                                 \
        }                                                                     \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerBinary<Op, SIMDType::NONE, SCALAR_VEC> {
    static void run(const typename Op::src_ctype src0,
                    const typename Op::src_ctype* src1,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType dst_dtype, size_t nr_elems) {
        Op op(src0_dtype, src1_dtype, dst_dtype);
        size_t i = 0;
        for (; i < nr_elems; i++) {
            op(src0, *src1, dst);
            src1++;
            dst++;
        }
    }
};
#undef OP_CALLER

#define OP_CALLER(simd_type, target_simd)                                     \
    template <typename Op>                                                    \
    struct OpCallerBinary<Op, simd_type, BCAST101_VEC> {                      \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                  \
        static void run(const typename Op::src_ctype* src0,                   \
                        const typename Op::src_ctype* src1,                   \
                        typename Op::dst_ctype* dst, DType src0_dtype,        \
                        DType src1_dtype, DType dst_dtype, size_t batch,      \
                        size_t channel, size_t channel_stride) {              \
            Op op(src0_dtype, src1_dtype, dst_dtype);                         \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis0;      \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis1;         \
            for (size_t b = 0; b < batch; b++) {                              \
                auto src0_ptr = src0;                                         \
                for (size_t c = 0; c < channel; c++) {                        \
                    auto vis0_simd = vis0(src0_ptr);                          \
                    size_t i = 0;                                             \
                    for (; i + Op::SIMD_WIDTH * 2 <= channel_stride;          \
                         i += Op::SIMD_WIDTH * 2) {                           \
                        op({{vis0_simd, vis0_simd}},                          \
                           {{vis1(src1), vis1(src1 + Op::SIMD_WIDTH)}}, dst); \
                        src1 += Op::SIMD_WIDTH * 2;                           \
                        dst += Op::SIMD_WIDTH * 2;                            \
                    }                                                         \
                    for (; i < channel_stride; i++) {                         \
                        op(*src0_ptr, *src1, dst);                            \
                        src1++;                                               \
                        dst++;                                                \
                    }                                                         \
                    src0_ptr++;                                               \
                }                                                             \
            }                                                                 \
        }                                                                     \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerBinary<Op, SIMDType::NONE, BCAST101_VEC> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType dst_dtype, size_t batch,
                    size_t channel, size_t channel_stride) {
        Op op(src0_dtype, src1_dtype, dst_dtype);
        for (size_t b = 0; b < batch; b++) {
            auto src0_ptr = src0;
            for (size_t c = 0; c < channel; c++) {
                for (size_t i = 0; i < channel_stride; i++) {
                    op(*src0_ptr, *src1, dst);
                    src1++;
                    dst++;
                }
                src0_ptr++;
            }
        }
    }
};
#undef OP_CALLER

template <typename Op>
struct OpCallerBinary<Op, SIMDType::AVX2, BCAST101x_VEC> {
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType dst_dtype, size_t batch,
                    size_t nr_channel_blocks, size_t channel_stride,
                    size_t channel_block_dim) {
        megdnn_assert(channel_block_dim == 8, "avx2 only support nchw88");
        Op op(src0_dtype, src1_dtype, dst_dtype);
        ParamElemVisitor<typename Op::src_ctype, SIMDType::AVX2> vis0;
        ParamElemVisitor<typename Op::src_ctype, SIMDType::AVX2> vis1;
        for (size_t b = 0; b < batch; b++) {
            auto src0_ptr = src0;
            for (size_t cb = 0; cb < nr_channel_blocks; cb++) {
                auto src0_block_ptr = src0_ptr + cb * channel_block_dim;
                auto channel_block_vec = vis0(src0_block_ptr);
                size_t img_index = 0;
                for (; img_index + 2 <= channel_stride; img_index += 2) {
                    op({{channel_block_vec, channel_block_vec}},
                       {{vis1(src1), vis1(src1 + Op::SIMD_WIDTH)}}, dst);
                    src1 += Op::SIMD_WIDTH * 2;
                    dst += Op::SIMD_WIDTH * 2;
                }
                for (; img_index < channel_stride; img_index++) {
                    for (size_t c_iter = 0; c_iter < channel_block_dim;
                         c_iter++) {
                        op(*(src0_block_ptr + c_iter), *src1, dst);
                        src1++;
                        dst++;
                    }
                }
            }
        }
    }
};

template <typename Op>
struct OpCallerBinary<Op, SIMDType::NONE, BCAST101x_VEC> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType dst_dtype, size_t batch,
                    size_t nr_channel_blocks, size_t channel_stride,
                    size_t channel_block_dim) {
        Op op(src0_dtype, src1_dtype, dst_dtype);
        for (size_t b = 0; b < batch; b++) {
            auto src0_ptr = src0;
            for (size_t cb = 0; cb < nr_channel_blocks; cb++) {
                auto src0_block_ptr = src0_ptr + cb * channel_block_dim;
                for (size_t i = 0; i < channel_stride; i++) {
                    for (size_t c_iter = 0; c_iter < channel_block_dim;
                         c_iter++) {
                        op(*(src0_block_ptr + c_iter), *src1, dst);
                        src1++;
                        dst++;
                    }
                }
            }
        }
    }
};

template <typename Op, SIMDType simd_type, BcastType bcast_type>
struct OpCallerTernary;

#define OP_CALLER(simd_type, target_simd)                                    \
    template <typename Op>                                                   \
    struct OpCallerTernary<Op, simd_type, VEC_VEC_VEC> {                     \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                 \
        static void run(const typename Op::src_ctype* src0,                  \
                        const typename Op::src_ctype* src1,                  \
                        const typename Op::src_ctype* src2,                  \
                        typename Op::dst_ctype* dst, DType src0_dtype,       \
                        DType src1_dtype, DType src2_dtype, DType dst_dtype, \
                        size_t nr_elems) {                                   \
            Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);            \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;        \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis1;        \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis2;        \
            size_t i = 0;                                                    \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                       \
                 i += Op::SIMD_WIDTH * 2) {                                  \
                op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},              \
                   {{vis1(src1), vis1(src1 + Op::SIMD_WIDTH)}},              \
                   {{vis2(src2), vis2(src2 + Op::SIMD_WIDTH)}}, dst);        \
                src0 += Op::SIMD_WIDTH * 2;                                  \
                src1 += Op::SIMD_WIDTH * 2;                                  \
                src2 += Op::SIMD_WIDTH * 2;                                  \
                dst += Op::SIMD_WIDTH * 2;                                   \
            }                                                                \
            for (; i < nr_elems; i++) {                                      \
                op(*src0, *src1, *src2, dst);                                \
                src0++;                                                      \
                src1++;                                                      \
                src2++;                                                      \
                dst++;                                                       \
            }                                                                \
        }                                                                    \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerTernary<Op, SIMDType::NONE, VEC_VEC_VEC> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    const typename Op::src_ctype* src2,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType src2_dtype, DType dst_dtype,
                    size_t nr_elems) {
        Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);
        size_t i = 0;
        for (; i < nr_elems; i++) {
            op(*src0, *src1, *src2, dst);
            src0++;
            src1++;
            src2++;
            dst++;
        }
    }
};
#undef OP_CALLER

//! src0: vector, src1: vector, src2: scalar
#define OP_CALLER(simd_type, target_simd)                                    \
    template <typename Op>                                                   \
    struct OpCallerTernary<Op, simd_type, VEC_VEC_SCALAR> {                  \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                 \
        static void run(const typename Op::src_ctype* src0,                  \
                        const typename Op::src_ctype* src1,                  \
                        const typename Op::src_ctype src2,                   \
                        typename Op::dst_ctype* dst, DType src0_dtype,       \
                        DType src1_dtype, DType src2_dtype, DType dst_dtype, \
                        size_t nr_elems) {                                   \
            Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);            \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;        \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis1;        \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis2;     \
            auto vis2_simd = vis2(&src2);                                    \
            size_t i = 0;                                                    \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                       \
                 i += Op::SIMD_WIDTH * 2) {                                  \
                op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},              \
                   {{vis1(src1), vis1(src1 + Op::SIMD_WIDTH)}},              \
                   {{vis2_simd, vis2_simd}}, dst);                           \
                src0 += Op::SIMD_WIDTH * 2;                                  \
                src1 += Op::SIMD_WIDTH * 2;                                  \
                dst += Op::SIMD_WIDTH * 2;                                   \
            }                                                                \
            for (; i < nr_elems; i++) {                                      \
                op(*src0, *src1, src2, dst);                                 \
                src0++;                                                      \
                src1++;                                                      \
                dst++;                                                       \
            }                                                                \
        }                                                                    \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerTernary<Op, SIMDType::NONE, VEC_VEC_SCALAR> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    const typename Op::src_ctype src2,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType src2_dtype, DType dst_dtype,
                    size_t nr_elems) {
        Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);
        for (size_t i = 0; i < nr_elems; i++) {
            op(*src0, *src1, src2, dst);
            src0++;
            src1++;
            dst++;
        }
    }
};
#undef OP_CALLER

//! src0: 1C11, src1: vector, src2:  1C11
#define OP_CALLER(simd_type, target_simd)                                     \
    template <typename Op>                                                    \
    struct OpCallerTernary<Op, simd_type, BCAST101_VEC_BCAST101> {            \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                  \
        static void run(const typename Op::src_ctype* src0,                   \
                        const typename Op::src_ctype* src1,                   \
                        const typename Op::src_ctype* src2,                   \
                        typename Op::dst_ctype* dst, DType src0_dtype,        \
                        DType src1_dtype, DType src2_dtype, DType dst_dtype,  \
                        size_t batch_size, size_t channel_size,               \
                        size_t channel_stride) {                              \
            Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);             \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis1;         \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis0;      \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis2;      \
            for (size_t batch = 0; batch < batch_size; batch++) {             \
                auto src0_ptr = src0;                                         \
                auto src2_ptr = src2;                                         \
                for (size_t channel = 0; channel < channel_size; channel++) { \
                    size_t i = 0;                                             \
                    auto src0_simd = vis0(src0_ptr);                          \
                    auto src2_simd = vis2(src2_ptr);                          \
                    for (; i + Op::SIMD_WIDTH * 2 <= channel_stride;          \
                         i += Op::SIMD_WIDTH * 2) {                           \
                        op({{src0_simd, src0_simd}},                          \
                           {{vis1(src1), vis1(src1 + Op::SIMD_WIDTH)}},       \
                           {{src2_simd, src2_simd}}, dst);                    \
                        src1 += Op::SIMD_WIDTH * 2;                           \
                        dst += Op::SIMD_WIDTH * 2;                            \
                    }                                                         \
                    for (; i < channel_stride; i++) {                         \
                        op(*src0_ptr, *src1, *src2_ptr, dst);                 \
                        src1++;                                               \
                        dst++;                                                \
                    }                                                         \
                    src0_ptr++;                                               \
                    src2_ptr++;                                               \
                }                                                             \
            }                                                                 \
        }                                                                     \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerTernary<Op, SIMDType::NONE, BCAST101_VEC_BCAST101> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    const typename Op::src_ctype* src2,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType src2_dtype, DType dst_dtype,
                    size_t batch_size, size_t channel_size,
                    size_t channel_stride) {
        Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);
        for (size_t batch = 0; batch < batch_size; batch++) {
            auto src0_ptr = src0;
            auto src2_ptr = src2;
            for (size_t channel = 0; channel < channel_size; channel++) {
                size_t i = 0;
                for (; i < channel_stride; i++) {
                    op(*src0_ptr, *src1, *src2_ptr, dst);
                    src1++;
                    dst++;
                }
                src0_ptr++;
                src2_ptr++;
            }
        }
    }
};
#undef OP_CALLER

//! src1: 1C11, src0 and src2 are contig
#define OP_CALLER(simd_type, target_simd)                                     \
    template <typename Op>                                                    \
    struct OpCallerTernary<Op, simd_type, VEC_BCAST101_VEC> {                 \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                  \
        static void run(const typename Op::src_ctype* src0,                   \
                        const typename Op::src_ctype* src1,                   \
                        const typename Op::src_ctype* src2,                   \
                        typename Op::dst_ctype* dst, DType src0_dtype,        \
                        DType src1_dtype, DType src2_dtype, DType dst_dtype,  \
                        size_t batch_size, size_t channel_size,               \
                        size_t channel_stride) {                              \
            Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);             \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;         \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis1;      \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis2;         \
            for (size_t batch = 0; batch < batch_size; batch++) {             \
                auto src1_ptr = src1;                                         \
                for (size_t channel = 0; channel < channel_size; channel++) { \
                    size_t i = 0;                                             \
                    auto src1_simd = vis1(src1_ptr);                          \
                    for (; i + Op::SIMD_WIDTH * 2 <= channel_stride;          \
                         i += Op::SIMD_WIDTH * 2) {                           \
                        op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},       \
                           {{src1_simd, src1_simd}},                          \
                           {{vis2(src2), vis2(src2 + Op::SIMD_WIDTH)}}, dst); \
                        src0 += Op::SIMD_WIDTH * 2;                           \
                        src2 += Op::SIMD_WIDTH * 2;                           \
                        dst += Op::SIMD_WIDTH * 2;                            \
                    }                                                         \
                    for (; i < channel_stride; i++) {                         \
                        op(*src0, *src1_ptr, *src2, dst);                     \
                        src0++;                                               \
                        src2++;                                               \
                        dst++;                                                \
                    }                                                         \
                    src1_ptr++;                                               \
                }                                                             \
            }                                                                 \
        }                                                                     \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerTernary<Op, SIMDType::NONE, VEC_BCAST101_VEC> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype* src1,
                    const typename Op::src_ctype* src2,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType src2_dtype, DType dst_dtype,
                    size_t batch_size, size_t channel_size,
                    size_t channel_stride) {
        Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);
        for (size_t batch = 0; batch < batch_size; batch++) {
            auto src1_ptr = src1;
            for (size_t channel = 0; channel < channel_size; channel++) {
                size_t i = 0;
                for (; i < channel_stride; i++) {
                    op(*src0, *src1_ptr, *src2, dst);
                    src0++;
                    src2++;
                    dst++;
                }
                src1_ptr++;
            }
        }
    }
};
#undef OP_CALLER

//! src1: scalar, src0 and src2 has the same shape
#define OP_CALLER(simd_type, target_simd)                                    \
    template <typename Op>                                                   \
    struct OpCallerTernary<Op, simd_type, VEC_SCALAR_VEC> {                  \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                 \
        static void run(const typename Op::src_ctype* src0,                  \
                        const typename Op::src_ctype src1,                   \
                        const typename Op::src_ctype* src2,                  \
                        typename Op::dst_ctype* dst, DType src0_dtype,       \
                        DType src1_dtype, DType src2_dtype, DType dst_dtype, \
                        size_t nr_elems) {                                   \
            Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);            \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;        \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis1;     \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis2;        \
            auto vis1_simd = vis1(&src1);                                    \
            size_t i = 0;                                                    \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                       \
                 i += Op::SIMD_WIDTH * 2) {                                  \
                op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},              \
                   {{vis1_simd, vis1_simd}},                                 \
                   {{vis2(src2), vis2(src2 + Op::SIMD_WIDTH)}}, dst);        \
                src0 += Op::SIMD_WIDTH * 2;                                  \
                src2 += Op::SIMD_WIDTH * 2;                                  \
                dst += Op::SIMD_WIDTH * 2;                                   \
            }                                                                \
            for (; i < nr_elems; i++) {                                      \
                op(*src0, src1, *src2, dst);                                 \
                src0++;                                                      \
                src2++;                                                      \
                dst++;                                                       \
            }                                                                \
        }                                                                    \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerTernary<Op, SIMDType::NONE, VEC_SCALAR_VEC> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype src1,
                    const typename Op::src_ctype* src2,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType src2_dtype, DType dst_dtype,
                    size_t nr_elems) {
        Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);
        size_t i = 0;
        for (; i < nr_elems; i++) {
            op(*src0, src1, *src2, dst);
            src0++;
            src2++;
            dst++;
        }
    }
};
#undef OP_CALLER

//! src1, src2: scalar, src0 is vector
#define OP_CALLER(simd_type, target_simd)                                    \
    template <typename Op>                                                   \
    struct OpCallerTernary<Op, simd_type, VEC_SCALAR_SCALAR> {               \
        MEGDNN_ATTRIBUTE_TARGET(target_simd)                                 \
        static void run(const typename Op::src_ctype* src0,                  \
                        const typename Op::src_ctype src1,                   \
                        const typename Op::src_ctype src2,                   \
                        typename Op::dst_ctype* dst, DType src0_dtype,       \
                        DType src1_dtype, DType src2_dtype, DType dst_dtype, \
                        size_t nr_elems) {                                   \
            Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);            \
            ParamElemVisitor<typename Op::src_ctype, simd_type> vis0;        \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis1;     \
            ParamElemVisitorDup<typename Op::src_ctype, simd_type> vis2;     \
            auto vis1_simd = vis1(&src1);                                    \
            auto vis2_simd = vis2(&src2);                                    \
            size_t i = 0;                                                    \
            for (; i + Op::SIMD_WIDTH * 2 <= nr_elems;                       \
                 i += Op::SIMD_WIDTH * 2) {                                  \
                op({{vis0(src0), vis0(src0 + Op::SIMD_WIDTH)}},              \
                   {{vis1_simd, vis1_simd}}, {{vis2_simd, vis2_simd}}, dst); \
                src0 += Op::SIMD_WIDTH * 2;                                  \
                dst += Op::SIMD_WIDTH * 2;                                   \
            }                                                                \
            for (; i < nr_elems; i++) {                                      \
                op(*src0, src1, src2, dst);                                  \
                src0++;                                                      \
                dst++;                                                       \
            }                                                                \
        }                                                                    \
    };
OP_CALLER(SIMDType::SSE4_2, "sse4.2")
OP_CALLER(SIMDType::AVX2, "avx2")

template <typename Op>
struct OpCallerTernary<Op, SIMDType::NONE, VEC_SCALAR_SCALAR> {
    static void run(const typename Op::src_ctype* src0,
                    const typename Op::src_ctype src1,
                    const typename Op::src_ctype src2,
                    typename Op::dst_ctype* dst, DType src0_dtype,
                    DType src1_dtype, DType src2_dtype, DType dst_dtype,
                    size_t nr_elems) {
        Op op(src0_dtype, src1_dtype, src2_dtype, dst_dtype);
        size_t i = 0;
        for (; i < nr_elems; i++) {
            op(*src0, src1, src2, dst);
            src0++;
            dst++;
        }
    }
};
#undef OP_CALLER

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
