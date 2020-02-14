/**
 * \file dnn/src/x86/elemwise_helper/kimpl/op_ternary_base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <immintrin.h>
#include "src/common/utils.h"
#include "src/x86/elemwise_helper/kimpl/op_unary_base.h"
#include "src/x86/quantized_converter.h"
#include "src/x86/simd_macro/immintrin.h"
#include "src/x86/utils.h"

namespace megdnn {
namespace x86 {

#define CONVERT_8_INT32_SSE(_type)                                         \
    __m128i val0_0 = _mm_cvtep##_type##_epi32(vsrc0);                      \
    __m128i val0_1 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc0, 4));  \
    __m128i val0_2 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc0, 8));  \
    __m128i val0_3 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc0, 12)); \
    __m128i val1_0 = _mm_cvtep##_type##_epi32(vsrc1);                      \
    __m128i val1_1 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc1, 4));  \
    __m128i val1_2 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc1, 8));  \
    __m128i val1_3 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc1, 12)); \
    __m128i val2_0 = _mm_cvtep##_type##_epi32(vsrc2);                      \
    __m128i val2_1 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc2, 4));  \
    __m128i val2_2 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc2, 8));  \
    __m128i val2_3 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc2, 12));

#define CONVERT_INT32_F32(_func_prefix)                   \
    auto fval0_0 = _##_func_prefix##_cvtepi32_ps(val0_0); \
    auto fval0_1 = _##_func_prefix##_cvtepi32_ps(val0_1); \
    auto fval0_2 = _##_func_prefix##_cvtepi32_ps(val0_2); \
    auto fval0_3 = _##_func_prefix##_cvtepi32_ps(val0_3); \
    auto fval1_0 = _##_func_prefix##_cvtepi32_ps(val1_0); \
    auto fval1_1 = _##_func_prefix##_cvtepi32_ps(val1_1); \
    auto fval1_2 = _##_func_prefix##_cvtepi32_ps(val1_2); \
    auto fval1_3 = _##_func_prefix##_cvtepi32_ps(val1_3); \
    auto fval2_0 = _##_func_prefix##_cvtepi32_ps(val2_0); \
    auto fval2_1 = _##_func_prefix##_cvtepi32_ps(val2_1); \
    auto fval2_2 = _##_func_prefix##_cvtepi32_ps(val2_2); \
    auto fval2_3 = _##_func_prefix##_cvtepi32_ps(val2_3);

#define CONVERT_8_INT32_AVX(_type)                                            \
    auto tmp0_0 = _mm256_extracti128_si256(vsrc0, 0);                         \
    auto tmp0_1 = _mm256_extracti128_si256(vsrc0, 1);                         \
    __m256i val0_0 = _mm256_cvtep##_type##_epi32(tmp0_0);                     \
    __m256i val0_1 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp0_0, 8)); \
    __m256i val0_2 = _mm256_cvtep##_type##_epi32(tmp0_1);                     \
    __m256i val0_3 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp0_1, 8)); \
    auto tmp1_0 = _mm256_extracti128_si256(vsrc1, 0);                         \
    auto tmp1_1 = _mm256_extracti128_si256(vsrc1, 1);                         \
    __m256i val1_0 = _mm256_cvtep##_type##_epi32(tmp1_0);                     \
    __m256i val1_1 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp1_0, 8)); \
    __m256i val1_2 = _mm256_cvtep##_type##_epi32(tmp1_1);                     \
    __m256i val1_3 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp1_1, 8)); \
    auto tmp2_0 = _mm256_extracti128_si256(vsrc2, 0);                         \
    auto tmp2_1 = _mm256_extracti128_si256(vsrc2, 1);                         \
    __m256i val2_0 = _mm256_cvtep##_type##_epi32(tmp2_0);                     \
    __m256i val2_1 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp2_0, 8)); \
    __m256i val2_2 = _mm256_cvtep##_type##_epi32(tmp2_1);                     \
    __m256i val2_3 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp2_1, 8));

////////////////////////// ternary //////////////////////////
template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct TernaryOpBase : OpBase<src_ctype, dst_ctype> {
    using OpBase<src_ctype, dst_ctype>::OpBase;
    TernaryOpBase() = default;
    TernaryOpBase(DType /*src0_dtype*/, DType /*src1_dtype*/,
                  DType /*src2_dtype*/, DType /*dst_dtype*/) {}
};

//////////////////////// quantization common ////////////////////

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)     \
    template <>                                                              \
    struct TernaryOpBase<_simd_type, dt_qint8, dt_qint8>                     \
            : OpBase<dt_qint8, dt_qint8> {                                   \
        using OpBase::OpBase;                                                \
        using src_ctype = dt_qint8;                                          \
        using dst_ctype = dt_qint8;                                          \
        float m_scale_src0, m_scale_src1, m_scale_src2, m_scale_dst;         \
        _simd_data_type m_vscale_src0, m_vscale_src1, m_vscale_src2,         \
                m_vscale_dst;                                                \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                \
        void init(float src0_scale, float src1_scale, float src2_scale,      \
                  float dst_scale) {                                         \
            m_scale_src0 = src0_scale;                                       \
            m_vscale_src0 = _##_func_prefix##_set1_ps(m_scale_src0);         \
            m_scale_src1 = src1_scale;                                       \
            m_vscale_src1 = _##_func_prefix##_set1_ps(m_scale_src1);         \
            m_scale_src2 = src2_scale;                                       \
            m_vscale_src2 = _##_func_prefix##_set1_ps(m_scale_src2);         \
            m_scale_dst = 1.f / dst_scale;                                   \
            m_vscale_dst = _##_func_prefix##_set1_ps(m_scale_dst);           \
        }                                                                    \
        TernaryOpBase(DType src0_dtype, DType src1_dtype, DType src2_dtype,  \
                      DType dst_dtype) {                                     \
            float src0_scale = src0_dtype.param<dtype::QuantizedS8>().scale; \
            float src1_scale = src1_dtype.param<dtype::QuantizedS8>().scale; \
            float src2_scale = src2_dtype.param<dtype::QuantizedS8>().scale; \
            float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;   \
            init(src0_scale, src1_scale, src2_scale, dst_scale);             \
        }                                                                    \
        TernaryOpBase(float src0_scale, float src1_scale, float src2_scale,  \
                      float dst_scale) {                                     \
            init(src0_scale, src1_scale, src2_scale, dst_scale);             \
        }                                                                    \
    };

OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct TernaryOpBase<_simd_type, dt_quint8, dt_quint8>                     \
            : OpBase<dt_quint8, dt_quint8> {                                   \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_quint8;                                           \
        using dst_ctype = dt_quint8;                                           \
        float m_scale_src0, m_scale_src1, m_scale_src2, m_scale_dst;           \
        _simd_data_type m_vscale_src0, m_vscale_src1, m_vscale_src2,           \
                m_vscale_dst;                                                  \
        uint8_t m_zp_src0, m_zp_src1, m_zp_src2, m_zp_dst;                     \
        _simd_data_type##i m_vzp_src0, m_vzp_src1, m_vzp_src2, m_vzp_dst;      \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float src0_scale, float src1_scale, float src2_scale,        \
                  float dst_scale, uint8_t src0_zp, uint8_t src1_zp,           \
                  uint8_t src2_zp, uint8_t dst_zp) {                           \
            m_scale_src0 = src0_scale;                                         \
            m_vscale_src0 = _##_func_prefix##_set1_ps(m_scale_src0);           \
            m_scale_src1 = src1_scale;                                         \
            m_vscale_src1 = _##_func_prefix##_set1_ps(m_scale_src1);           \
            m_scale_src2 = src2_scale;                                         \
            m_vscale_src2 = _##_func_prefix##_set1_ps(m_scale_src2);           \
            m_scale_dst = 1.f / dst_scale;                                     \
            m_vscale_dst = _##_func_prefix##_set1_ps(m_scale_dst);             \
            m_zp_src0 = src0_zp;                                               \
            m_zp_src1 = src1_zp;                                               \
            m_zp_src2 = src2_zp;                                               \
            m_zp_dst = dst_zp;                                                 \
            m_vzp_src0 = _##_func_prefix##_set1_epi32(m_zp_src0);              \
            m_vzp_src1 = _##_func_prefix##_set1_epi32(m_zp_src1);              \
            m_vzp_src2 = _##_func_prefix##_set1_epi32(m_zp_src2);              \
            m_vzp_dst = _##_func_prefix##_set1_epi32(m_zp_dst);                \
        }                                                                      \
        TernaryOpBase(DType src0_dtype, DType src1_dtype, DType src2_dtype,    \
                      DType dst_dtype) {                                       \
            float src0_scale =                                                 \
                    src0_dtype.param<dtype::Quantized8Asymm>().scale;          \
            float src1_scale =                                                 \
                    src1_dtype.param<dtype::Quantized8Asymm>().scale;          \
            float src2_scale =                                                 \
                    src2_dtype.param<dtype::Quantized8Asymm>().scale;          \
            float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale; \
            uint8_t src0_zp =                                                  \
                    src0_dtype.param<dtype::Quantized8Asymm>().zero_point;     \
            uint8_t src1_zp =                                                  \
                    src1_dtype.param<dtype::Quantized8Asymm>().zero_point;     \
            uint8_t src2_zp =                                                  \
                    src2_dtype.param<dtype::Quantized8Asymm>().zero_point;     \
            uint8_t dst_zp =                                                   \
                    dst_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(src0_scale, src1_scale, src2_scale, dst_scale, src0_zp,       \
                 src1_zp, src2_zp, dst_zp);                                    \
        }                                                                      \
        TernaryOpBase(float src0_scale, float src1_scale, float src2_scale,    \
                      float dst_scale, uint8_t src0_zp, uint8_t src1_zp,       \
                      uint8_t src2_zp, uint8_t dst_zp) {                       \
            init(src0_scale, src1_scale, src2_scale, dst_scale, src0_zp,       \
                 src1_zp, src2_zp, dst_zp);                                    \
        }                                                                      \
    };

OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

template <SIMDType simd_type, typename src_type, typename dst_type, typename Op>
struct TernaryQuantizationOp;

//! because gcc<9 get the class menber of type __m256 will bring a compile
//! error! just like this: internal compiler error: in convert_move, at
//! expr.c:315

#define SUB_MUL_TO_F32(_func_prefix, val, zp, scale)                        \
    _##_func_prefix##_mul_ps(_##_func_prefix##_cvtepi32_ps(                 \
                                     _##_func_prefix##_sub_epi32(val, zp)), \
                             scale);

#define OPERATE(_func_prefix, scale_dst)                  \
    auto vitem0 = op(vitem0_0, vitem1_0, vitem2_0);       \
    auto vitem1 = op(vitem0_1, vitem1_1, vitem2_1);       \
    auto vitem2 = op(vitem0_2, vitem1_2, vitem2_2);       \
    auto vitem3 = op(vitem0_3, vitem1_3, vitem2_3);       \
    vitem0 = _##_func_prefix##_mul_ps(vitem0, scale_dst); \
    vitem1 = _##_func_prefix##_mul_ps(vitem1, scale_dst); \
    vitem2 = _##_func_prefix##_mul_ps(vitem2, scale_dst); \
    vitem3 = _##_func_prefix##_mul_ps(vitem3, scale_dst);

#define ALL_MUL_SCALE(_func_prefix, _scale)                       \
    auto vitem0_0 = _##_func_prefix##_mul_ps(fval0_0, _scale##0); \
    auto vitem0_1 = _##_func_prefix##_mul_ps(fval0_1, _scale##0); \
    auto vitem0_2 = _##_func_prefix##_mul_ps(fval0_2, _scale##0); \
    auto vitem0_3 = _##_func_prefix##_mul_ps(fval0_3, _scale##0); \
    auto vitem1_0 = _##_func_prefix##_mul_ps(fval1_0, _scale##1); \
    auto vitem1_1 = _##_func_prefix##_mul_ps(fval1_1, _scale##1); \
    auto vitem1_2 = _##_func_prefix##_mul_ps(fval1_2, _scale##1); \
    auto vitem1_3 = _##_func_prefix##_mul_ps(fval1_3, _scale##1); \
    auto vitem2_0 = _##_func_prefix##_mul_ps(fval2_0, _scale##2); \
    auto vitem2_1 = _##_func_prefix##_mul_ps(fval2_1, _scale##2); \
    auto vitem2_2 = _##_func_prefix##_mul_ps(fval2_2, _scale##2); \
    auto vitem2_3 = _##_func_prefix##_mul_ps(fval2_3, _scale##2);

#define ALL_SUB_ZERO_MUL_SCALE(_prefix, _vzp, _scale)                    \
    auto vitem0_0 = SUB_MUL_TO_F32(_prefix, val0_0, _vzp##0, _scale##0); \
    auto vitem0_1 = SUB_MUL_TO_F32(_prefix, val0_1, _vzp##0, _scale##0); \
    auto vitem0_2 = SUB_MUL_TO_F32(_prefix, val0_2, _vzp##0, _scale##0); \
    auto vitem0_3 = SUB_MUL_TO_F32(_prefix, val0_3, _vzp##0, _scale##0); \
    auto vitem1_0 = SUB_MUL_TO_F32(_prefix, val1_0, _vzp##1, _scale##1); \
    auto vitem1_1 = SUB_MUL_TO_F32(_prefix, val1_1, _vzp##1, _scale##1); \
    auto vitem1_2 = SUB_MUL_TO_F32(_prefix, val1_2, _vzp##1, _scale##1); \
    auto vitem1_3 = SUB_MUL_TO_F32(_prefix, val1_3, _vzp##1, _scale##1); \
    auto vitem2_0 = SUB_MUL_TO_F32(_prefix, val2_0, _vzp##2, _scale##2); \
    auto vitem2_1 = SUB_MUL_TO_F32(_prefix, val2_1, _vzp##2, _scale##2); \
    auto vitem2_2 = SUB_MUL_TO_F32(_prefix, val2_2, _vzp##2, _scale##2); \
    auto vitem2_3 = SUB_MUL_TO_F32(_prefix, val2_3, _vzp##2, _scale##2);

#define OPERATOR_TERNARY_QINT8_SSE() \
    ALL_MUL_SCALE(mm, m_vscale_src)  \
    OPERATE(mm, m_vscale_dst)

#define OPERATOR_TERNARY_QINT8_AVX()                 \
    auto vscale_src0 = _mm256_set1_ps(m_scale_src0); \
    auto vscale_src1 = _mm256_set1_ps(m_scale_src1); \
    auto vscale_src2 = _mm256_set1_ps(m_scale_src2); \
    auto vscale_dst = _mm256_set1_ps(m_scale_dst);   \
    ALL_MUL_SCALE(mm256, vscale_src)                 \
    OPERATE(mm256, vscale_dst)

#define OPERATOR_TERNARY_QUINT8_SSE()                   \
    ALL_SUB_ZERO_MUL_SCALE(mm, m_vzp_src, m_vscale_src) \
    OPERATE(mm, m_vscale_dst)

#define OPERATOR_TERNARY_QUINT8_AVX()                  \
    auto vscale_src0 = _mm256_set1_ps(m_scale_src0);   \
    auto vscale_src1 = _mm256_set1_ps(m_scale_src1);   \
    auto vscale_src2 = _mm256_set1_ps(m_scale_src2);   \
    auto vscale_dst = _mm256_set1_ps(m_scale_dst);     \
    auto vzp_src0 = _mm256_set1_epi32(m_zp_src0);      \
    auto vzp_src1 = _mm256_set1_epi32(m_zp_src1);      \
    auto vzp_src2 = _mm256_set1_epi32(m_zp_src2);      \
    ALL_SUB_ZERO_MUL_SCALE(mm256, vzp_src, vscale_src) \
    OPERATE(mm256, vscale_dst)

template <typename Op>
struct TernaryQuantizationOp<SIMDType::SSE4_2, dt_qint8, dt_qint8, Op>
        : TernaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_qint8> {
    using TernaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_qint8>::TernaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;
    void operator()(const dt_qint8& src0, const dt_qint8& src1,
                    const dt_qint8& src2, dt_qint8* dst) const {
        *dst = operator()(src0, src1, src2);
    }
    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1,
                        const dt_qint8& src2) const {
        float fsrc0 = src0.as_int8() * m_scale_src0;
        float fsrc1 = src1.as_int8() * m_scale_src1;
        float fsrc2 = src2.as_int8() * m_scale_src2;
        float fsrc = op(fsrc0, fsrc1, fsrc2);
        fsrc = fsrc * m_scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    void operator()(const __m128ix2& vsrc0, const __m128ix2& vsrc1,
                    const __m128ix2& vsrc2, dt_qint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc0.val[0],
                                                            vsrc1.val[0],
                                                            vsrc2.val[0]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + SIMD_WIDTH),
                         operator()(vsrc0.val[1], vsrc1.val[1], vsrc2.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    __m128i operator()(const __m128i& vsrc0, const __m128i& vsrc1,
                       const __m128i& vsrc2) const {
        CONVERT_8_INT32_SSE(i8)
        CONVERT_INT32_F32(mm)
        OPERATOR_TERNARY_QINT8_SSE()
        auto result0 =
                QConverter::convert<int64_t, __m128x2>({{vitem0, vitem1}});
        auto result1 =
                QConverter::convert<int64_t, __m128x2>({{vitem2, vitem3}});
        return _mm_set_epi64x(result1, result0);
    }
};

template <typename Op>
struct TernaryQuantizationOp<SIMDType::AVX2, dt_qint8, dt_qint8, Op>
        : TernaryOpBase<SIMDType::AVX2, dt_qint8, dt_qint8> {
    using TernaryOpBase<SIMDType::AVX2, dt_qint8, dt_qint8>::TernaryOpBase;
    constexpr static size_t SIMD_WIDTH = 32;
    Op op;
    void operator()(const dt_qint8& src0, const dt_qint8& src1,
                    const dt_qint8& src2, dt_qint8* dst) const {
        *dst = operator()(src0, src1, src2);
    }
    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1,
                        const dt_qint8& src2) const {
        float fsrc0 = src0.as_int8() * m_scale_src0;
        float fsrc1 = src1.as_int8() * m_scale_src1;
        float fsrc2 = src2.as_int8() * m_scale_src2;
        float fsrc = op(fsrc0, fsrc1, fsrc2);
        fsrc = fsrc * m_scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    void operator()(const __m256ix2& vsrc0, const __m256ix2& vsrc1,
                    const __m256ix2& vsrc2, dt_qint8* dst) const {
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst), operator()(vsrc0.val[0],
                                                            vsrc1.val[0],
                                                            vsrc2.val[0]));
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst + SIMD_WIDTH),
                operator()(vsrc0.val[1], vsrc1.val[1], vsrc2.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    __m256i operator()(const __m256i& vsrc0, const __m256i& vsrc1,
                       const __m256i& vsrc2) const {
        CONVERT_8_INT32_AVX(i8)
        CONVERT_INT32_F32(mm256)
        OPERATOR_TERNARY_QINT8_AVX()
        auto result0 =
                QConverter::convert<__m128i, __m256x2>({{vitem0, vitem1}});
        auto result1 =
                QConverter::convert<__m128i, __m256x2>({{vitem2, vitem3}});
        return _mm256_set_m128i(result1, result0);
    }
};
template <typename Op>
struct TernaryQuantizationOp<SIMDType::SSE4_2, dt_quint8, dt_quint8, Op>
        : TernaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_quint8> {
    using TernaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_quint8>::TernaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_quint8& src0, const dt_quint8& src1,
                    const dt_quint8& src2, dt_quint8* dst) const {
        *dst = operator()(src0, src1, src2);
    }
    dt_quint8 operator()(const dt_quint8& src0, const dt_quint8& src1,
                         const dt_quint8& src2) const {
        float fsrc0 = (src0.as_uint8() - m_zp_src0) * m_scale_src0;
        float fsrc1 = (src1.as_uint8() - m_zp_src1) * m_scale_src1;
        float fsrc2 = (src2.as_uint8() - m_zp_src2) * m_scale_src2;
        float fsrc = op(fsrc0, fsrc1, fsrc2);
        fsrc = fsrc * m_scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, m_zp_dst);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    void operator()(const __m128ix2& vsrc0, const __m128ix2& vsrc1,
                    const __m128ix2& vsrc2, dt_quint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc0.val[0],
                                                            vsrc1.val[0],
                                                            vsrc2.val[0]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + SIMD_WIDTH),
                         operator()(vsrc0.val[1], vsrc1.val[1], vsrc2.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    __m128i operator()(const __m128i& vsrc0, const __m128i& vsrc1,
                       const __m128i& vsrc2) const {
        CONVERT_8_INT32_SSE(u8)
        OPERATOR_TERNARY_QUINT8_SSE()
        auto result0 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem0, vitem1}}, m_vzp_dst);
        auto result1 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem2, vitem3}}, m_vzp_dst);
        return _mm_set_epi64x(result1, result0);
    }
};

template <typename Op>
struct TernaryQuantizationOp<SIMDType::AVX2, dt_quint8, dt_quint8, Op>
        : TernaryOpBase<SIMDType::AVX2, dt_quint8, dt_quint8> {
    using TernaryOpBase<SIMDType::AVX2, dt_quint8, dt_quint8>::TernaryOpBase;
    constexpr static size_t SIMD_WIDTH = 32;
    Op op;
    void operator()(const dt_quint8& src0, const dt_quint8& src1,
                    const dt_quint8& src2, dt_quint8* dst) const {
        *dst = operator()(src0, src1, src2);
    }
    dt_quint8 operator()(const dt_quint8& src0, const dt_quint8& src1,
                         const dt_quint8& src2) const {
        float fsrc0 = (src0.as_uint8() - m_zp_src0) * m_scale_src0;
        float fsrc1 = (src1.as_uint8() - m_zp_src1) * m_scale_src1;
        float fsrc2 = (src2.as_uint8() - m_zp_src2) * m_scale_src2;
        float fsrc = op(fsrc0, fsrc1, fsrc2);
        fsrc = fsrc * m_scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, m_zp_dst);
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    void operator()(const __m256ix2& vsrc0, const __m256ix2& vsrc1,
                    const __m256ix2& vsrc2, dt_quint8* dst) const {
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst), operator()(vsrc0.val[0],
                                                            vsrc1.val[0],
                                                            vsrc2.val[0]));
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst + SIMD_WIDTH),
                operator()(vsrc0.val[1], vsrc1.val[1], vsrc2.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    __m256i operator()(const __m256i& vsrc0, const __m256i& vsrc1,
                       const __m256i& vsrc2) const {
        CONVERT_8_INT32_AVX(u8)
        OPERATOR_TERNARY_QUINT8_AVX()
        auto v_dzp = _mm256_set1_epi32(m_zp_dst);
        auto result0 = QConverter::convert<__m128i, __m256x2, __m256i>(
                {{vitem0, vitem1}}, v_dzp);
        auto result1 = QConverter::convert<__m128i, __m256x2, __m256i>(
                {{vitem2, vitem3}}, v_dzp);
        return _mm256_set_m128i(result1, result0);
    }
};
#undef ALL_MUL_SCALE
#undef ALL_SUB_ZERO_MUL_SCALE
#undef OPERATE
#undef SUB_MUL_TO_F32

#undef OPERATOR_TERNARY_QUINT8_SSE
#undef OPERATOR_TERNARY_QUINT8_AVX
#undef OPERATOR_TERNARY_QUINT8_SSE
#undef OPERATOR_TERNARY_QUINT8_AVX

#undef CONVERT_8_INT32_SSE
#undef CONVERT_INT32_F32
#undef CONVERT_8_INT32_AVX
}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
