/**
 * \file dnn/src/x86/elemwise_helper/kimpl/op_unary_base.h
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
#include "src/x86/quantized_converter.h"
#include "src/x86/simd_macro/immintrin.h"
#include "src/x86/utils.h"

namespace megdnn {
namespace x86 {
#define CONVERT_8_INT32_SSE(_type)                                      \
    __m128i val_0 = _mm_cvtep##_type##_epi32(vsrc);                     \
    __m128i val_1 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc, 4)); \
    __m128i val_2 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc, 8)); \
    __m128i val_3 = _mm_cvtep##_type##_epi32(_mm_bsrli_si128(vsrc, 12));

#define CONVERT_INT32_F32(_func_prefix)                 \
    auto fval_0 = _##_func_prefix##_cvtepi32_ps(val_0); \
    auto fval_1 = _##_func_prefix##_cvtepi32_ps(val_1); \
    auto fval_2 = _##_func_prefix##_cvtepi32_ps(val_2); \
    auto fval_3 = _##_func_prefix##_cvtepi32_ps(val_3);

#define CONVERT_8_INT32_AVX(_type)                                         \
    auto tmp0 = _mm256_extracti128_si256(vsrc, 0);                         \
    auto tmp1 = _mm256_extracti128_si256(vsrc, 1);                         \
    __m256i val_0 = _mm256_cvtep##_type##_epi32(tmp0);                     \
    __m256i val_1 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp0, 8)); \
    __m256i val_2 = _mm256_cvtep##_type##_epi32(tmp1);                     \
    __m256i val_3 = _mm256_cvtep##_type##_epi32(_mm_bsrli_si128(tmp1, 8));

////////////////////////// unary //////////////////////////
template <typename _src_ctype, typename _dst_ctype = _src_ctype>
struct OpBase {
    using src_ctype = _src_ctype;
    using dst_ctype = _dst_ctype;
    OpBase() = default;
};

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct UnaryOpBase : OpBase<src_ctype, dst_ctype> {
    using OpBase<src_ctype, dst_ctype>::OpBase;
    UnaryOpBase() = default;
    UnaryOpBase(DType /*src_dtype*/, DType /*dst_dtype*/) {}
};

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct UnaryOpBase<_simd_type, dt_quint8, dt_quint8>                       \
            : OpBase<dt_quint8, dt_quint8> {                                   \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_quint8;                                           \
        using dst_ctype = dt_quint8;                                           \
        float scale, scale_src, scale_dst;                                     \
        uint8_t dzp, szp;                                                      \
        _simd_data_type vscale, vscale_src, vscale_dst;                        \
        _simd_data_type##i vszp, vdzp;                                         \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float src_scale, float dst_scale, uint8_t src_zp,            \
                  uint8_t dst_zp) {                                            \
            scale_src = src_scale;                                             \
            scale_dst = 1.f / dst_scale;                                       \
            scale = scale_src * scale_dst;                                     \
            vscale = _##_func_prefix##_set1_ps(scale);                         \
            vscale_src = _##_func_prefix##_set1_ps(scale_src);                 \
            vscale_dst = _##_func_prefix##_set1_ps(scale_dst);                 \
            dzp = dst_zp;                                                      \
            szp = src_zp;                                                      \
            vszp = _##_func_prefix##_set1_epi32(szp);                          \
            vdzp = _##_func_prefix##_set1_epi32(dzp);                          \
        }                                                                      \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                        \
            float src_scale = src_dtype.param<dtype::Quantized8Asymm>().scale; \
            float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale; \
            uint8_t src_zp =                                                   \
                    src_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            uint8_t dst_zp =                                                   \
                    dst_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(src_scale, dst_scale, src_zp, dst_zp);                        \
        }                                                                      \
        UnaryOpBase(float src_scale, float dst_scale, uint8_t src_zp,          \
                    uint8_t dst_zp) {                                          \
            init(src_scale, dst_scale, src_zp, dst_zp);                        \
        }                                                                      \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct UnaryOpBase<_simd_type, dt_qint8, dt_quint8>                        \
            : OpBase<dt_qint8, dt_quint8> {                                    \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_qint8;                                            \
        using dst_ctype = dt_quint8;                                           \
        float scale, scale_src, scale_dst;                                     \
        uint8_t dzp;                                                           \
        _simd_data_type vscale, vscale_src, vscale_dst;                        \
        _simd_data_type##i vdzp;                                               \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float src_scale, float dst_scale, uint8_t dst_zp) {          \
            scale_src = src_scale;                                             \
            scale_dst = 1.f / dst_scale;                                       \
            scale = scale_src * scale_dst;                                     \
            vscale = _##_func_prefix##_set1_ps(scale);                         \
            vscale_src = _##_func_prefix##_set1_ps(scale_src);                 \
            vscale_dst = _##_func_prefix##_set1_ps(scale_dst);                 \
            dzp = dst_zp;                                                      \
            vdzp = _##_func_prefix##_set1_epi32(dzp);                          \
        }                                                                      \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                        \
            float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;     \
            float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale; \
            uint8_t dst_zp =                                                   \
                    dst_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(src_scale, dst_scale, dst_zp);                                \
        }                                                                      \
        UnaryOpBase(float src_scale, float dst_scale, uint8_t dst_zp) {        \
            init(src_scale, dst_scale, dst_zp);                                \
        }                                                                      \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct UnaryOpBase<_simd_type, dt_qint32, dt_quint8>                       \
            : OpBase<dt_qint32, dt_quint8> {                                   \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_qint32;                                           \
        using dst_ctype = dt_quint8;                                           \
        float scale, scale_src, scale_dst;                                     \
        uint8_t dzp;                                                           \
        _simd_data_type vscale, vscale_src, vscale_dst;                        \
        _simd_data_type##i vdzp;                                               \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float src_scale, float dst_scale, uint8_t dst_zp) {          \
            scale_src = src_scale;                                             \
            scale_dst = 1.0f / dst_scale;                                      \
            dzp = dst_zp;                                                      \
            vdzp = _##_func_prefix##_set1_epi32(static_cast<int>(dzp));        \
            scale = src_scale / dst_scale;                                     \
            vscale = _##_func_prefix##_set1_ps(scale);                         \
            vscale_src = _##_func_prefix##_set1_ps(scale_src);                 \
            vscale_dst = _##_func_prefix##_set1_ps(scale_dst);                 \
        }                                                                      \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                        \
            float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;    \
            float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale; \
            uint8_t dst_zp =                                                   \
                    dst_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(src_scale, dst_scale, dst_zp);                                \
        }                                                                      \
        UnaryOpBase(float src_scale, float dst_scale, uint8_t dst_zp) {        \
            init(src_scale, dst_scale, dst_zp);                                \
        }                                                                      \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct UnaryOpBase<_simd_type, dt_float32, dt_quint8>                      \
            : OpBase<dt_float32, dt_quint8> {                                  \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_float32;                                          \
        using dst_ctype = dt_quint8;                                           \
        float scale;                                                           \
        uint8_t dzp;                                                           \
        _simd_data_type vscale;                                                \
        _simd_data_type##i vdzp;                                               \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float dst_scale, uint8_t dst_zp) {                           \
            dzp = dst_zp;                                                      \
            vdzp = _##_func_prefix##_set1_epi32(static_cast<int>(dzp));        \
            scale = 1.0f / dst_scale;                                          \
            vscale = _##_func_prefix##_set1_ps(scale);                         \
        }                                                                      \
        UnaryOpBase(DType, DType dst_dtype) {                                  \
            float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale; \
            uint8_t dst_zp =                                                   \
                    dst_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(dst_scale, dst_zp);                                           \
        }                                                                      \
        UnaryOpBase(float dst_scale, uint8_t dst_zp) {                         \
            init(dst_scale, dst_zp);                                           \
        }                                                                      \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)   \
    template <>                                                            \
    struct UnaryOpBase<_simd_type, dt_qint8, dt_qint8>                     \
            : OpBase<dt_qint8, dt_qint8> {                                 \
        using OpBase::OpBase;                                              \
        using src_ctype = dt_qint8;                                        \
        using dst_ctype = dt_qint8;                                        \
        float scale, scale_src, scale_dst;                                 \
        _simd_data_type vscale, vscale_src, vscale_dst;                    \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                              \
        void init(float src_scale, float dst_scale) {                      \
            scale_src = src_scale;                                         \
            scale_dst = 1.0f / dst_scale;                                  \
            scale = src_scale / dst_scale;                                 \
            vscale = _##_func_prefix##_set1_ps(scale);                     \
            vscale_src = _##_func_prefix##_set1_ps(scale_src);             \
            vscale_dst = _##_func_prefix##_set1_ps(scale_dst);             \
        }                                                                  \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                    \
            float src_scale = src_dtype.param<dtype::QuantizedS8>().scale; \
            float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale; \
            init(src_scale, dst_scale);                                    \
        }                                                                  \
        UnaryOpBase(float src_scale, float dst_scale) {                    \
            init(src_scale, dst_scale);                                    \
        }                                                                  \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct UnaryOpBase<_simd_type, dt_quint8, dt_qint8>                        \
            : OpBase<dt_quint8, dt_qint8> {                                    \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_quint8;                                           \
        using dst_ctype = dt_qint8;                                            \
        float scale, scale_src, scale_dst;                                     \
        uint8_t szp;                                                           \
        _simd_data_type vscale, vscale_src, vscale_dst;                        \
        _simd_data_type##i vszp;                                               \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float src_scale, float dst_scale, uint8_t src_zp) {          \
            scale_src = src_scale;                                             \
            scale_dst = 1.f / dst_scale;                                       \
            scale = scale_src * scale_dst;                                     \
            vscale = _##_func_prefix##_set1_ps(scale);                         \
            szp = src_zp;                                                      \
            vszp = _##_func_prefix##_set1_epi32(szp);                          \
            vscale_src = _##_func_prefix##_set1_ps(scale_src);                 \
            vscale_dst = _##_func_prefix##_set1_ps(scale_dst);                 \
        }                                                                      \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                        \
            float src_scale = src_dtype.param<dtype::Quantized8Asymm>().scale; \
            float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;     \
            uint8_t src_zp =                                                   \
                    src_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(src_scale, dst_scale, src_zp);                                \
        }                                                                      \
        UnaryOpBase(float src_scale, float dst_scale, uint8_t src_zp) {        \
            init(src_scale, dst_scale, src_zp);                                \
        }                                                                      \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)    \
    template <>                                                             \
    struct UnaryOpBase<_simd_type, dt_qint32, dt_qint8>                     \
            : OpBase<dt_qint32, dt_qint8> {                                 \
        using OpBase::OpBase;                                               \
        using src_ctype = dt_qint32;                                        \
        using dst_ctype = dt_qint8;                                         \
        float scale, scale_src, scale_dst;                                  \
        _simd_data_type vscale, vscale_src, vscale_dst;                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        void init(float src_scale, float dst_scale) {                       \
            scale_src = src_scale;                                          \
            scale_dst = 1.f / dst_scale;                                    \
            scale = src_scale / dst_scale;                                  \
            vscale = _##_func_prefix##_set1_ps(scale);                      \
            vscale_src = _##_func_prefix##_set1_ps(scale_src);              \
            vscale_dst = _##_func_prefix##_set1_ps(scale_dst);              \
        }                                                                   \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                     \
            float src_scale = src_dtype.param<dtype::QuantizedS32>().scale; \
            float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;  \
            init(src_scale, dst_scale);                                     \
        }                                                                   \
        UnaryOpBase(float src_scale, float dst_scale) {                     \
            init(src_scale, dst_scale);                                     \
        }                                                                   \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

template <>
struct UnaryOpBase<SIMDType::NONE, dt_qint32, dt_qint8>
        : OpBase<dt_qint32, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_qint8;
    float scale, scale_src, scale_dst;
    void init(float src_scale, float dst_scale) {
        scale_src = src_scale;
        scale_dst = 1.f / dst_scale;
        scale = src_scale / dst_scale;
    }
    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src_scale, dst_scale);
    }
    UnaryOpBase(float src_scale, float dst_scale) {
        init(src_scale, dst_scale);
    }
};

template <>
struct UnaryOpBase<SIMDType::NONE, dt_qint32, dt_quint8>
        : OpBase<dt_qint32, dt_quint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_quint8;
    float scale, scale_src, scale_dst;
    uint8_t dzp;
    void init(float src_scale, float dst_scale, uint8_t dst_zp) {
        scale_src = src_scale;
        scale_dst = 1.0f / dst_scale;
        dzp = dst_zp;
        scale = src_scale / dst_scale;
    }
    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        uint8_t dst_zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        init(src_scale, dst_scale, dst_zp);
    }
    UnaryOpBase(float src_scale, float dst_scale, uint8_t dst_zp) {
        init(src_scale, dst_scale, dst_zp);
    }
};

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)   \
    template <>                                                            \
    struct UnaryOpBase<_simd_type, dt_float32, dt_qint8>                   \
            : OpBase<dt_float32, dt_qint8> {                               \
        using OpBase::OpBase;                                              \
        using src_ctype = dt_float32;                                      \
        using dst_ctype = dt_qint8;                                        \
        float scale;                                                       \
        _simd_data_type vscale;                                            \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                              \
        void init(float dst_scale) {                                       \
            scale = 1.0f / dst_scale;                                      \
            vscale = _##_func_prefix##_set1_ps(scale);                     \
        }                                                                  \
        UnaryOpBase(DType, DType dst_dtype) {                              \
            float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale; \
            init(dst_scale);                                               \
        }                                                                  \
        UnaryOpBase(float dst_scale) { init(dst_scale); }                  \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)    \
    template <>                                                             \
    struct UnaryOpBase<_simd_type, dt_qint8, dt_qint32>                     \
            : OpBase<dt_qint8, dt_qint32> {                                 \
        using OpBase::OpBase;                                               \
        using src_ctype = dt_qint8;                                         \
        using dst_ctype = dt_qint32;                                        \
        float scale, scale_src, scale_dst;                                  \
        _simd_data_type vscale, vscale_src, vscale_dst;                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        void init(float src_scale, float dst_scale) {                       \
            scale_src = src_scale;                                          \
            scale_dst = 1.f / dst_scale;                                    \
            scale = src_scale / dst_scale;                                  \
            vscale = _##_func_prefix##_set1_ps(scale);                      \
            vscale_src = _##_func_prefix##_set1_ps(src_scale);              \
            vscale_dst = _##_func_prefix##_set1_ps(1.0f / dst_scale);       \
        }                                                                   \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                     \
            float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;  \
            float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale; \
            init(src_scale, dst_scale);                                     \
        }                                                                   \
        UnaryOpBase(float src_scale, float dst_scale) {                     \
            init(src_scale, dst_scale);                                     \
        }                                                                   \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct UnaryOpBase<_simd_type, dt_quint8, dt_qint32>                       \
            : OpBase<dt_quint8, dt_qint32> {                                   \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_quint8;                                           \
        using dst_ctype = dt_qint32;                                           \
        float scale, scale_src, scale_dst;                                     \
        uint8_t szp;                                                           \
        _simd_data_type vscale, vscale_src, vscale_dst;                        \
        _simd_data_type##i vszp;                                               \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float src_scale, float dst_scale, uint8_t src_zp) {          \
            scale_src = src_scale;                                             \
            scale_dst = 1.f / dst_scale;                                       \
            scale = scale_src * scale_dst;                                     \
            vscale = _##_func_prefix##_set1_ps(scale);                         \
            vscale_src = _##_func_prefix##_set1_ps(scale_src);                 \
            vscale_dst = _##_func_prefix##_set1_ps(scale_dst);                 \
            szp = src_zp;                                                      \
            vszp = _##_func_prefix##_set1_epi32(szp);                          \
        }                                                                      \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                        \
            float src_scale = src_dtype.param<dtype::Quantized8Asymm>().scale; \
            float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale;    \
            uint8_t src_zp =                                                   \
                    src_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(src_scale, dst_scale, src_zp);                                \
        }                                                                      \
        UnaryOpBase(float src_scale, float dst_scale, uint8_t src_zp) {        \
            init(src_scale, dst_scale, src_zp);                                \
        }                                                                      \
    };
OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)    \
    template <>                                                             \
    struct UnaryOpBase<_simd_type, dt_qint32, dt_qint32>                    \
            : OpBase<dt_qint32, dt_qint32> {                                \
        using OpBase::OpBase;                                               \
        using src_ctype = dt_qint32;                                        \
        using dst_ctype = dt_qint32;                                        \
        float scale, scale_src, scale_dst;                                  \
        _simd_data_type vscale, vscale_src, vscale_dst;                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        void init(float src_scale, float dst_scale) {                       \
            scale_src = src_scale;                                          \
            scale_dst = 1.f / dst_scale;                                    \
            scale = src_scale / dst_scale;                                  \
            vscale = _##_func_prefix##_set1_ps(scale);                      \
            vscale_src = _##_func_prefix##_set1_ps(src_scale);              \
            vscale_dst = _##_func_prefix##_set1_ps(1.0f / dst_scale);       \
        }                                                                   \
        UnaryOpBase(DType src_dtype, DType dst_dtype) {                     \
            float src_scale = src_dtype.param<dtype::QuantizedS32>().scale; \
            float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale; \
            init(src_scale, dst_scale);                                     \
        }                                                                   \
        UnaryOpBase(float src_scale, float dst_scale) {                     \
            init(src_scale, dst_scale);                                     \
        }                                                                   \
    };

OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)    \
    template <>                                                             \
    struct UnaryOpBase<_simd_type, dt_float32, dt_qint32>                   \
            : OpBase<dt_float32, dt_qint32> {                               \
        using OpBase::OpBase;                                               \
        using src_ctype = dt_float32;                                       \
        using dst_ctype = dt_qint32;                                        \
        float scale;                                                        \
        _simd_data_type vscale;                                             \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        void init(float dst_scale) {                                        \
            scale = 1.0f / dst_scale;                                       \
            vscale = _##_func_prefix##_set1_ps(scale);                      \
        }                                                                   \
        UnaryOpBase(DType, DType dst_dtype) {                               \
            float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale; \
            init(dst_scale);                                                \
        }                                                                   \
        UnaryOpBase(float dst_scale) { init(dst_scale); }                   \
    };

OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)   \
    template <>                                                            \
    struct UnaryOpBase<_simd_type, dt_qint8, dt_float32>                   \
            : OpBase<dt_qint8, dt_float32> {                               \
        using OpBase::OpBase;                                              \
        using src_ctype = dt_qint8;                                        \
        using dst_ctype = dt_float32;                                      \
        float scale;                                                       \
        _simd_data_type vscale;                                            \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                              \
        void init(float src_scale) {                                       \
            scale = src_scale;                                             \
            vscale = _##_func_prefix##_set1_ps(scale);                     \
        }                                                                  \
        UnaryOpBase(DType src_dtype, DType) {                              \
            float src_scale = src_dtype.param<dtype::QuantizedS8>().scale; \
            init(src_scale);                                               \
        }                                                                  \
        UnaryOpBase(float src_scale) { init(src_scale); }                  \
    };

OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)       \
    template <>                                                                \
    struct UnaryOpBase<_simd_type, dt_quint8, dt_float32>                      \
            : OpBase<dt_quint8, dt_float32> {                                  \
        using OpBase::OpBase;                                                  \
        using src_ctype = dt_quint8;                                           \
        using dst_ctype = dt_float32;                                          \
        float scale;                                                           \
        uint8_t szp;                                                           \
        _simd_data_type vscale;                                                \
        _simd_data_type##i vszp;                                               \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void init(float src_scale, uint8_t src_zp) {                           \
            float scale_src = src_scale;                                       \
            scale = scale_src;                                                 \
            vscale = _##_func_prefix##_set1_ps(scale);                         \
            szp = src_zp;                                                      \
            vszp = _##_func_prefix##_set1_epi32(szp);                          \
        }                                                                      \
        UnaryOpBase(DType src_dtype, DType) {                                  \
            float src_scale = src_dtype.param<dtype::Quantized8Asymm>().scale; \
            uint8_t src_zp =                                                   \
                    src_dtype.param<dtype::Quantized8Asymm>().zero_point;      \
            init(src_scale, src_zp);                                           \
        }                                                                      \
        UnaryOpBase(float src_scale, uint8_t src_zp) {                         \
            init(src_scale, src_zp);                                           \
        }                                                                      \
    };

OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

#define OP_BASE(_simd_type, _simd_target, _simd_data_type, _func_prefix)    \
    template <>                                                             \
    struct UnaryOpBase<_simd_type, dt_qint32, dt_float32>                   \
            : OpBase<dt_qint32, dt_float32> {                               \
        using OpBase::OpBase;                                               \
        using src_ctype = dt_qint32;                                        \
        using dst_ctype = dt_float32;                                       \
        float scale;                                                        \
        _simd_data_type vscale;                                             \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        void init(float src_scale) {                                        \
            scale = src_scale;                                              \
            vscale = _##_func_prefix##_set1_ps(scale);                      \
        }                                                                   \
        UnaryOpBase(DType src_dtype, DType) {                               \
            float src_scale = src_dtype.param<dtype::QuantizedS32>().scale; \
            init(src_scale);                                                \
        }                                                                   \
        UnaryOpBase(float src_scale) { init(src_scale); }                   \
    };

OP_BASE(SIMDType::SSE4_2, "sse4.2", __m128, mm)
OP_BASE(SIMDType::AVX2, "avx2", __m256, mm256)
#undef OP_BASE

//////////////////////// quantization common ////////////////////
template <SIMDType simd_type, typename src_type, typename dst_type, typename Op>
struct UnaryQuantizationOp;
//! because gcc<9 get the class menber of type __m256 will bring a compile
//! error! just like this: internal compiler error: in convert_move, at
//! expr.c:315
//
#define OPERATOR_UNARY_QINT8(_func_prefix)                       \
    auto v_scale_src = _##_func_prefix##_set1_ps(scale_src);     \
    auto v_scale_dst = _##_func_prefix##_set1_ps(scale_dst);     \
    auto vitem0 = _##_func_prefix##_mul_ps(fval_0, v_scale_src); \
    auto vitem1 = _##_func_prefix##_mul_ps(fval_1, v_scale_src); \
    auto vitem2 = _##_func_prefix##_mul_ps(fval_2, v_scale_src); \
    auto vitem3 = _##_func_prefix##_mul_ps(fval_3, v_scale_src); \
    vitem0 = op(vitem0);                                         \
    vitem1 = op(vitem1);                                         \
    vitem2 = op(vitem2);                                         \
    vitem3 = op(vitem3);                                         \
    vitem0 = _##_func_prefix##_mul_ps(vitem0, v_scale_dst);      \
    vitem1 = _##_func_prefix##_mul_ps(vitem1, v_scale_dst);      \
    vitem2 = _##_func_prefix##_mul_ps(vitem2, v_scale_dst);      \
    vitem3 = _##_func_prefix##_mul_ps(vitem3, v_scale_dst);

#define OPERATOR_UNARY_QUINT8(_func_prefix)                     \
    auto v_scale_src = _##_func_prefix##_set1_ps(scale_src);    \
    auto v_scale_dst = _##_func_prefix##_set1_ps(scale_dst);    \
    auto v_szp = _##_func_prefix##_set1_epi32(szp);             \
    auto vitem0 = _##_func_prefix##_mul_ps(                     \
            _##_func_prefix##_cvtepi32_ps(                      \
                    _##_func_prefix##_sub_epi32(val_0, v_szp)), \
            v_scale_src);                                       \
    auto vitem1 = _##_func_prefix##_mul_ps(                     \
            _##_func_prefix##_cvtepi32_ps(                      \
                    _##_func_prefix##_sub_epi32(val_1, v_szp)), \
            v_scale_src);                                       \
    auto vitem2 = _##_func_prefix##_mul_ps(                     \
            _##_func_prefix##_cvtepi32_ps(                      \
                    _##_func_prefix##_sub_epi32(val_2, v_szp)), \
            v_scale_src);                                       \
    auto vitem3 = _##_func_prefix##_mul_ps(                     \
            _##_func_prefix##_cvtepi32_ps(                      \
                    _##_func_prefix##_sub_epi32(val_3, v_szp)), \
            v_scale_src);                                       \
    vitem0 = op(vitem0);                                        \
    vitem1 = op(vitem1);                                        \
    vitem2 = op(vitem2);                                        \
    vitem3 = op(vitem3);                                        \
    vitem0 = _##_func_prefix##_mul_ps(vitem0, v_scale_dst);     \
    vitem1 = _##_func_prefix##_mul_ps(vitem1, v_scale_dst);     \
    vitem2 = _##_func_prefix##_mul_ps(vitem2, v_scale_dst);     \
    vitem3 = _##_func_prefix##_mul_ps(vitem3, v_scale_dst);

template <typename Op>
struct UnaryQuantizationOp<SIMDType::SSE4_2, dt_qint8, dt_qint8, Op>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_qint8> {
    using UnaryOpBase<SIMDType::SSE4_2, dt_qint8, dt_qint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_qint8& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }

    dt_qint8 operator()(const dt_qint8& src) const {
        float fsrc = src.as_int8() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }

    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    void operator()(const __m128ix2& vsrc, dt_qint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + SIMD_WIDTH),
                         operator()(vsrc.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    __m128i operator()(const __m128i& vsrc) const {
        CONVERT_8_INT32_SSE(i8)
        CONVERT_INT32_F32(mm)
        OPERATOR_UNARY_QINT8(mm)
        auto result0 =
                QConverter::convert<int64_t, __m128x2>({{vitem0, vitem1}});
        auto result1 =
                QConverter::convert<int64_t, __m128x2>({{vitem2, vitem3}});
        return _mm_set_epi64x(result1, result0);
    }
};

template <typename Op>
struct UnaryQuantizationOp<SIMDType::AVX2, dt_qint8, dt_qint8, Op>
        : UnaryOpBase<SIMDType::AVX2, dt_qint8, dt_qint8> {
    using UnaryOpBase<SIMDType::AVX2, dt_qint8, dt_qint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 32;
    Op op;

    void operator()(const dt_qint8& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }

    dt_qint8 operator()(const dt_qint8& src) const {
        float fsrc = src.as_int8() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }

    MEGDNN_ATTRIBUTE_TARGET("avx2")
    void operator()(const __m256ix2& vsrc, dt_qint8* dst) const {
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst), operator()(vsrc.val[0]));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + SIMD_WIDTH),
                            operator()(vsrc.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    __m256i operator()(const __m256i& vsrc) const {
        CONVERT_8_INT32_AVX(i8)
        CONVERT_INT32_F32(mm256)
        OPERATOR_UNARY_QINT8(mm256)
        auto result0 =
                QConverter::convert<__m128i, __m256x2>({{vitem0, vitem1}});
        auto result1 =
                QConverter::convert<__m128i, __m256x2>({{vitem2, vitem3}});
        return _mm256_set_m128i(result1, result0);
    }
};
template <typename Op>
struct UnaryQuantizationOp<SIMDType::SSE4_2, dt_quint8, dt_quint8, Op>
        : UnaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_quint8> {
    using UnaryOpBase<SIMDType::SSE4_2, dt_quint8, dt_quint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_quint8& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }
    dt_quint8 operator()(const dt_quint8& src) const {
        float fsrc = (src.as_uint8() - szp) * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    void operator()(const __m128ix2& vsrc, dt_quint8* dst) const {
        _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst), operator()(vsrc.val[0]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + SIMD_WIDTH),
                         operator()(vsrc.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    __m128i operator()(const __m128i& vsrc) const {
        CONVERT_8_INT32_SSE(u8)
        OPERATOR_UNARY_QUINT8(mm)
        auto result0 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem0, vitem1}}, this->vdzp);
        auto result1 = QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem2, vitem3}}, this->vdzp);
        return _mm_set_epi64x(result1, result0);
    }
};

template <typename Op>
struct UnaryQuantizationOp<SIMDType::AVX2, dt_quint8, dt_quint8, Op>
        : UnaryOpBase<SIMDType::AVX2, dt_quint8, dt_quint8> {
    using UnaryOpBase<SIMDType::AVX2, dt_quint8, dt_quint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 32;
    Op op;

    void operator()(const dt_quint8& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }
    dt_quint8 operator()(const dt_quint8& src) const {
        float fsrc = (src.as_uint8() - szp) * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    void operator()(const __m256ix2& vsrc, dt_quint8* dst) const {
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst), operator()(vsrc.val[0]));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + SIMD_WIDTH),
                            operator()(vsrc.val[1]));
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    __m256i operator()(const __m256i& vsrc) const {
        CONVERT_8_INT32_AVX(u8)
        OPERATOR_UNARY_QUINT8(mm256)
        auto v_dzp = _mm256_set1_epi32(dzp);
        auto result0 = QConverter::convert<__m128i, __m256x2, __m256i>(
                {{vitem0, vitem1}}, v_dzp);
        auto result1 = QConverter::convert<__m128i, __m256x2, __m256i>(
                {{vitem2, vitem3}}, v_dzp);
        return _mm256_set_m128i(result1, result0);
    }
};
template <typename Op>
struct UnaryQuantizationOp<SIMDType::NONE, dt_qint32, dt_qint8, Op>
        : UnaryOpBase<SIMDType::NONE, dt_qint32, dt_qint8> {
    using UnaryOpBase<SIMDType::NONE, dt_qint32, dt_qint8>::UnaryOpBase;
    Op op;
    void operator()(const dt_qint32& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }
    dt_qint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
};

template <typename Op>
struct UnaryQuantizationOp<SIMDType::SSE4_2, dt_qint32, dt_qint8, Op>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_qint8> {
    using UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_qint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;
    Op op;
    void operator()(const dt_qint32& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }
    dt_qint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    void operator()(const __m128ix2& vsrc, dt_qint8* dst) const {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst),
                         _mm_set1_epi64x(operator()(vsrc)));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    int64_t operator()(const __m128ix2& vsrc) const {
        auto vitem0 =
                _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[0]), this->vscale_src);
        auto vitem1 =
                _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[1]), this->vscale_src);
        vitem0 = op(vitem0);
        vitem1 = op(vitem1);
        vitem0 = _mm_mul_ps(vitem0, this->vscale_dst);
        vitem1 = _mm_mul_ps(vitem1, this->vscale_dst);
        return QConverter::convert<int64_t, __m128x2>({{vitem0, vitem1}});
    }
};

template <typename Op>
struct UnaryQuantizationOp<SIMDType::NONE, dt_qint32, dt_quint8, Op>
        : UnaryOpBase<SIMDType::NONE, dt_qint32, dt_quint8> {
    using UnaryOpBase<SIMDType::NONE, dt_qint32, dt_quint8>::UnaryOpBase;
    Op op;

    void operator()(const dt_qint32& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }
    dt_quint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }
};

template <typename Op>
struct UnaryQuantizationOp<SIMDType::AVX2, dt_qint32, dt_qint8, Op>
        : UnaryOpBase<SIMDType::AVX2, dt_qint32, dt_qint8> {
    using UnaryOpBase<SIMDType::AVX2, dt_qint32, dt_qint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 8;
    Op op;
    void operator()(const dt_qint32& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }
    dt_qint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    void operator()(const __m256ix2& vsrc, dt_qint8* dst) const {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), operator()(vsrc));
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    __m128i operator()(const __m256ix2& vsrc) const {
        auto v_scale_src = _mm256_set1_ps(scale_src);
        auto v_scale_dst = _mm256_set1_ps(scale_dst);
        auto vitem0 =
                _mm256_mul_ps(_mm256_cvtepi32_ps(vsrc.val[0]), v_scale_src);
        auto vitem1 =
                _mm256_mul_ps(_mm256_cvtepi32_ps(vsrc.val[1]), v_scale_src);
        vitem0 = op(vitem0);
        vitem1 = op(vitem1);
        vitem0 = _mm256_mul_ps(vitem0, v_scale_dst);
        vitem1 = _mm256_mul_ps(vitem1, v_scale_dst);
        return QConverter::convert<__m128i, __m256x2>({{vitem0, vitem1}});
    }
};

template <typename Op>
struct UnaryQuantizationOp<SIMDType::SSE4_2, dt_qint32, dt_quint8, Op>
        : UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_quint8> {
    using UnaryOpBase<SIMDType::SSE4_2, dt_qint32, dt_quint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 8;
    Op op;

    void operator()(const dt_qint32& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }
    dt_quint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    void operator()(const __m128ix2& vsrc, dt_quint8* dst) const {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst),
                         _mm_set1_epi64x(operator()(vsrc)));
    }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    int64_t operator()(const __m128ix2& vsrc) const {
        auto vitem0 =
                _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[0]), this->vscale_src);
        auto vitem1 =
                _mm_mul_ps(_mm_cvtepi32_ps(vsrc.val[1]), this->vscale_src);
        vitem0 = op(vitem0);
        vitem1 = op(vitem1);
        vitem0 = _mm_mul_ps(vitem0, this->vscale_dst);
        vitem1 = _mm_mul_ps(vitem1, this->vscale_dst);
        return QConverter::convert<int64_t, __m128x2, __m128i>(
                {{vitem0, vitem1}}, this->vdzp);
    }
};

template <typename Op>
struct UnaryQuantizationOp<SIMDType::AVX2, dt_qint32, dt_quint8, Op>
        : UnaryOpBase<SIMDType::AVX2, dt_qint32, dt_quint8> {
    using UnaryOpBase<SIMDType::AVX2, dt_qint32, dt_quint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 8;
    Op op;

    void operator()(const dt_qint32& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }
    dt_quint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    void operator()(const __m256ix2& vsrc, dt_quint8* dst) const {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), operator()(vsrc));
    }

    MEGDNN_ATTRIBUTE_TARGET("avx2")
    __m128i operator()(const __m256ix2& vsrc) const {
        auto v_scale_src = _mm256_set1_ps(scale_src);
        auto v_scale_dst = _mm256_set1_ps(scale_dst);
        auto v_dzp = _mm256_set1_epi32(dzp);
        auto vitem0 =
                _mm256_mul_ps(_mm256_cvtepi32_ps(vsrc.val[0]), v_scale_src);
        auto vitem1 =
                _mm256_mul_ps(_mm256_cvtepi32_ps(vsrc.val[1]), v_scale_src);
        vitem0 = op(vitem0);
        vitem1 = op(vitem1);
        vitem0 = _mm256_mul_ps(vitem0, v_scale_dst);
        vitem1 = _mm256_mul_ps(vitem1, v_scale_dst);
        return QConverter::convert<__m128i, __m256x2, __m256i>(
                {{vitem0, vitem1}}, v_dzp);
    }
};

#undef OPERATOR_UNARY_QINT8
#undef OPERATOR_UNARY_QUINT8

#undef CONVERT_8_INT32_SSE
#undef CONVERT_INT32_F32
#undef CONVERT_8_INT32_AVX

}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
