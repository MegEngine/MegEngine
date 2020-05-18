/**
 * \file dnn/src/x86/elemwise_helper/kimpl/typecvt.h
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
#ifdef WIN32
#include <avx2intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <smmintrin.h>
#endif
#include "src/common/utils.h"
#include "src/x86/elemwise_helper/kimpl/op_unary_base.h"
#include "src/x86/quantized_converter.h"
#include "src/x86/utils.h"

namespace megdnn {
namespace x86 {

#define CONVERT_INT8_INT32                                       \
    __m128i val_0 = _mm_cvtepi8_epi32(vsrc);                     \
    __m128i val_1 = _mm_cvtepi8_epi32(_mm_bsrli_si128(vsrc, 4)); \
    __m128i val_2 = _mm_cvtepi8_epi32(_mm_bsrli_si128(vsrc, 8)); \
    __m128i val_3 = _mm_cvtepi8_epi32(_mm_bsrli_si128(vsrc, 12));

#define CONVERT_UINT8_INT32                                      \
    __m128i val_0 = _mm_cvtepu8_epi32(vsrc);                     \
    __m128i val_1 = _mm_cvtepu8_epi32(_mm_bsrli_si128(vsrc, 4)); \
    __m128i val_2 = _mm_cvtepu8_epi32(_mm_bsrli_si128(vsrc, 8)); \
    __m128i val_3 = _mm_cvtepu8_epi32(_mm_bsrli_si128(vsrc, 12));

#define CONVERT_INT32_F32                   \
    __m128 fval_0 = _mm_cvtepi32_ps(val_0); \
    __m128 fval_1 = _mm_cvtepi32_ps(val_1); \
    __m128 fval_2 = _mm_cvtepi32_ps(val_2); \
    __m128 fval_3 = _mm_cvtepi32_ps(val_3);

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct TypeCvtOp;

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_quint8, dt_quint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_quint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));
        dst += SIMD_WIDTH;
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[1]));
    }

    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128i operator()(const __m128i& vsrc) const {
        CONVERT_UINT8_INT32
        auto vitem0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_0, vszp)),
                                 this->vscale);
        auto vitem1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_1, vszp)),
                                 this->vscale);
        auto vitem2 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_2, vszp)),
                                 this->vscale);
        auto vitem3 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_3, vszp)),
                                 this->vscale);
        auto result0 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem0, vitem1}}, this->vdzp);
        auto result1 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem2, vitem3}}, this->vdzp);
        return _mm_set_epi64x(result1, result0);
    }

    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<uint8_t*>(dst) = saturate<uint8_t, float>(
                std::round((src.as_uint8() - szp) * scale) + dzp, 0, 255);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint8, dt_quint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_quint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));
        dst += SIMD_WIDTH;
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128i operator()(const __m128i& vsrc) const {
        CONVERT_INT8_INT32
        CONVERT_INT32_F32
        auto vitem0 = _mm_mul_ps(fval_0, this->vscale);
        auto vitem1 = _mm_mul_ps(fval_1, this->vscale);
        auto vitem2 = _mm_mul_ps(fval_2, this->vscale);
        auto vitem3 = _mm_mul_ps(fval_3, this->vscale);
        auto result0 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem0, vitem1}}, this->vdzp);
        auto result1 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem2, vitem3}}, this->vdzp);
        return _mm_set_epi64x(result1, result0);
    }

    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<uint8_t*>(dst) = saturate<uint8_t, float>(
                std::round(src.as_int8() * scale) + dzp, 0, 255);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint32, dt_quint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const __m128ix2& vsrc, dt_quint8* dst) const {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst),
                         _mm_set1_epi64x(operator()(vsrc)));
    }
    int64_t operator()(const __m128ix2& vsrc) const {
        auto vitem0 = _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[0]), this->vscale);
        auto vitem1 = _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[1]), this->vscale);
        auto result = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem0, vitem1}}, this->vdzp);
        return result;
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<uint8_t*>(dst) = saturate<uint8_t, float>(
                std::round(src.as_int32() * scale) + dzp, 0, 255);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_float32, dt_quint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_float32, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const __m128x2& vsrc, dt_quint8* dst) const {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst),
                         _mm_set1_epi64x(operator()(vsrc)));
    }
    int64_t operator()(const __m128x2& vsrc) const {
        auto vitem0 = _mm_mul_ps(vsrc.val[0], this->vscale);
        auto vitem1 = _mm_mul_ps(vsrc.val[1], this->vscale);
        return QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem0, vitem1}}, this->vdzp);
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<uint8_t*>(dst) =
                saturate<uint8_t, float>(std::round(src * scale) + dzp, 0, 255);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint8, dt_qint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_qint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));

        dst += SIMD_WIDTH;
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128i operator()(const __m128i& vsrc) const {
        CONVERT_INT8_INT32
        CONVERT_INT32_F32
        auto vitem0 = _mm_mul_ps(fval_0, this->vscale);
        auto vitem1 = _mm_mul_ps(fval_1, this->vscale);
        auto vitem2 = _mm_mul_ps(fval_2, this->vscale);
        auto vitem3 = _mm_mul_ps(fval_3, this->vscale);
        auto result0 =
                QConverter::convert<int64_t, __m128x2>({{vitem0, vitem1}});
        auto result1 =
                QConverter::convert<int64_t, __m128x2>({{vitem2, vitem3}});
        return _mm_set_epi64x(result1, result0);
    }

    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<uint8_t*>(dst) = saturate<int8_t, float>(
                std::round(src.as_int8() * scale), -128, 127);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_quint8, dt_qint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_qint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));
        dst += SIMD_WIDTH;
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128i operator()(const __m128i& vsrc) const {
        CONVERT_UINT8_INT32
        auto vitem0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_0, vszp)),
                                 this->vscale);
        auto vitem1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_1, vszp)),
                                 this->vscale);
        auto vitem2 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_2, vszp)),
                                 this->vscale);
        auto vitem3 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_3, vszp)),
                                 this->vscale);
        auto result0 =
                QConverter::convert<int64_t, __m128x2>({{vitem0, vitem1}});
        auto result1 =
                QConverter::convert<int64_t, __m128x2>({{vitem2, vitem3}});
        return _mm_set_epi64x(result1, result0);
    }

    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int8_t*>(dst) = saturate<int8_t, float>(
                std::round((src.as_uint8() - szp) * scale), -128, 127);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint32, dt_qint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const __m128ix2& vsrc, dt_qint8* dst) const {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst),
                         _mm_set1_epi64x(operator()(vsrc)));
    }
    int64_t operator()(const __m128ix2& vsrc) const {
        auto vitem0 = _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[0]), this->vscale);
        auto vitem1 = _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[1]), this->vscale);
        return QConverter::convert<int64_t, __m128x2>({{vitem0, vitem1}});
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int8_t*>(dst) = saturate<int8_t, float>(
                std::round(src.as_int32() * scale), -128, 127);
    }
};

template <>
struct TypeCvtOp<SIMDType::AVX2, dt_qint32, dt_qint8>
        : UnaryOpBase<SIMDType::AVX2, dt_qint32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 8;

    MEGDNN_ATTRIBUTE_TARGET("avx2")
    void operator()(const __m256ix2& vsrc, dt_qint8* dst) const {
        _mm_store_si128((__m128i*)(dst), (operator()(vsrc)));
    }

    MEGDNN_ATTRIBUTE_TARGET("avx2")
    __m128i operator()(const __m256ix2& vsrc) const {
        auto cvtps_src0 = _mm256_cvtepi32_ps(vsrc.val[0]);
        auto cvtps_src1 = _mm256_cvtepi32_ps(vsrc.val[1]);
        auto vitem0 = _mm256_mul_ps(cvtps_src0, _mm256_set1_ps(this->scale));
        auto vitem1 = _mm256_mul_ps(cvtps_src1, _mm256_set1_ps(this->scale));
        return QConverter::convert<__m128i, __m256x2>({{vitem0, vitem1}});
    }

    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int8_t*>(dst) = saturate<int8_t, float>(
                std::round(src.as_int32() * scale), -128, 127);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_float32, dt_qint8>
        : UnaryOpBase<SIMDType::SSE4_2, dt_float32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const __m128x2 vsrc, dt_qint8* dst) const {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst),
                         _mm_set1_epi64x(operator()(vsrc)));
    }
    int64_t operator()(const __m128x2& vsrc) const {
        auto vitem0 = _mm_mul_ps(vsrc.val[0], this->vscale);
        auto vitem1 = _mm_mul_ps(vsrc.val[1], this->vscale);
        return QConverter::convert<int64_t, __m128x2>({{vitem0, vitem1}});
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int8_t*>(dst) =
                saturate<int8_t, float>(std::round(src * scale), -128, 127);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_quint8, dt_qint32>
        : UnaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_qint32> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_qint32* dst) const {
        auto result0 = operator()(vsrc.val[0]);
        auto result1 = operator()(vsrc.val[1]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[0]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[1]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[2]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[3]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[0]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[1]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[2]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[3]);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128ix4 operator()(const __m128i& vsrc) const {
        CONVERT_UINT8_INT32
        auto vitem0 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_0, this->vszp)),
                           this->vscale);
        auto vitem1 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_1, this->vszp)),
                           this->vscale);
        auto vitem2 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_2, this->vszp)),
                           this->vscale);
        auto vitem3 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_3, this->vszp)),
                           this->vscale);
        return {{QConverter::convert<__m128i, __m128>(vitem0),
                 QConverter::convert<__m128i, __m128>(vitem1),
                 QConverter::convert<__m128i, __m128>(vitem2),
                 QConverter::convert<__m128i, __m128>(vitem3)}};
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int32_t*>(dst) =
                std::round((src.as_uint8() - szp) * scale);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint8, dt_qint32>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_qint32> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_qint32* dst) const {
        auto result0 = operator()(vsrc.val[0]);
        auto result1 = operator()(vsrc.val[1]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[0]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[1]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[2]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result0.val[3]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[0]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[1]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[2]);
        dst += 4;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result1.val[3]);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128ix4 operator()(const __m128i& vsrc) const {
        CONVERT_INT8_INT32
        CONVERT_INT32_F32
        auto vitem0 = _mm_mul_ps(fval_0, this->vscale);
        auto vitem1 = _mm_mul_ps(fval_1, this->vscale);
        auto vitem2 = _mm_mul_ps(fval_2, this->vscale);
        auto vitem3 = _mm_mul_ps(fval_3, this->vscale);
        return {{QConverter::convert<__m128i, __m128>(vitem0),
                 QConverter::convert<__m128i, __m128>(vitem1),
                 QConverter::convert<__m128i, __m128>(vitem2),
                 QConverter::convert<__m128i, __m128>(vitem3)}};
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int32_t*>(dst) = std::round(src.as_int8() * scale);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint32, dt_qint32>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_qint32> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const __m128ix2& vsrc, dt_qint32* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));
        dst += SIMD_WIDTH;
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[1]));
    }
    __m128i operator()(const __m128i& vsrc) const {
        auto vitem0 = _mm_mul_ps(_mm_cvtepi32_ps(vsrc), this->vscale);
        return QConverter::convert<__m128i, __m128>(vitem0);
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int32_t*>(dst) = std::round(src.as_int32() * scale);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_float32, dt_qint32>
        : UnaryOpBase<SIMDType::SSE4_2, dt_float32, dt_qint32> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const __m128x2& vsrc, dt_qint32* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));
        dst += SIMD_WIDTH;
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[1]));
    }
    __m128i operator()(const __m128& vsrc) const {
        auto vitem0 = _mm_mul_ps(vsrc, this->vscale);
        return QConverter::convert<__m128i, __m128>(vitem0);
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<int32_t*>(dst) = std::round(src * scale);
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_quint8, dt_float32>
        : UnaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_float32> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_float32* dst) const {
        auto result0 = operator()(vsrc.val[0]);
        auto result1 = operator()(vsrc.val[1]);
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[0]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[1]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[2]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[3]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[0]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[1]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[2]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[3]);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128x4 operator()(const __m128i& vsrc) const {
        CONVERT_UINT8_INT32
        auto vitem0 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_0, this->vszp)),
                           this->vscale);
        auto vitem1 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_1, this->vszp)),
                           this->vscale);
        auto vitem2 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_2, this->vszp)),
                           this->vscale);
        auto vitem3 =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(val_3, this->vszp)),
                           this->vscale);
        return {{vitem0, vitem1, vitem2, vitem3}};
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<float*>(dst) = (src.as_uint8() - szp) * scale;
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint8, dt_float32>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_float32> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;

    void operator()(const __m128ix2& vsrc, dt_float32* dst) const {
        auto result0 = operator()(vsrc.val[0]);
        auto result1 = operator()(vsrc.val[1]);
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[0]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[1]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[2]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result0.val[3]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[0]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[1]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[2]);
        dst += 4;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), result1.val[3]);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.1")
    __m128x4 operator()(const __m128i& vsrc) const {
        CONVERT_INT8_INT32
        CONVERT_INT32_F32
        auto vitem0 = _mm_mul_ps(fval_0, this->vscale);
        auto vitem1 = _mm_mul_ps(fval_1, this->vscale);
        auto vitem2 = _mm_mul_ps(fval_2, this->vscale);
        auto vitem3 = _mm_mul_ps(fval_3, this->vscale);
        return {{vitem0, vitem1, vitem2, vitem3}};
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<float*>(dst) = src.as_int8() * scale;
    }
};

template <>
struct TypeCvtOp<SIMDType::SSE4_2, dt_qint32, dt_float32>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_float32> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const __m128ix2& vsrc, dt_float32* dst) const {
        _mm_storeu_ps(reinterpret_cast<float*>(dst), operator()(vsrc.val[0]));
        dst += SIMD_WIDTH;
        _mm_storeu_ps(reinterpret_cast<float*>(dst), operator()(vsrc.val[1]));
    }
    __m128 operator()(const __m128i& vsrc) const {
        return _mm_mul_ps(_mm_cvtepi32_ps(vsrc), this->vscale);
    }
    void operator()(src_ctype src, dst_ctype* dst) {
        *reinterpret_cast<float*>(dst) = src.as_int32() * scale;
    }
};

template <>
struct TypeCvtOp<SIMDType::NONE, dt_float32, dt_float32>
        : UnaryOpBase<SIMDType::NONE, dt_float32, dt_float32> {
    using UnaryOpBase::UnaryOpBase;

    float operator()(float& src) const { return src; }
};

#undef CONVERT_INT8_INT32
#undef CONVERT_UINT8_INT32
#undef CONVERT_INT32_F32
}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
