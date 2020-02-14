/**
 * \file dnn/src/x86/elemwise_helper/kimpl/abs.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/elemwise_helper/kimpl/op_unary_base.h"
#include "src/x86/quantized_converter.h"

namespace megdnn {
namespace x86 {

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct AbsOpBase : UnaryOpBase<simd_type, src_ctype, dst_ctype> {
    using UnaryOpBase<simd_type, src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        return src > 0 ? src : (-src);
    }
};

template <SIMDType simd_type, typename src_ctype, typename dst_type = src_ctype>
struct AbsOp;

#define OP(_ctype, _simd_type, _simd_target, _simd_data_type,                \
           _simd_data_type2, _func_prefix, _func_suffix1, _func_suffix2,     \
           _simd_width)                                                      \
    template <>                                                              \
    struct AbsOp<_simd_type, _ctype> : AbsOpBase<_simd_type, _ctype> {       \
        using AbsOpBase::AbsOpBase;                                          \
        using AbsOpBase::operator();                                         \
        constexpr static size_t SIMD_WIDTH = _simd_width;                    \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                \
        void operator()(const _simd_data_type2& src, _ctype* dst) const {    \
            auto vitem = operator()(src);                                    \
            _##_func_prefix##_storeu_##_func_suffix2(                        \
                    reinterpret_cast<_simd_data_type*>(dst), vitem.val[0]);  \
            _##_func_prefix##_storeu_##_func_suffix2(                        \
                    reinterpret_cast<_simd_data_type*>(dst + SIMD_WIDTH),    \
                    vitem.val[1]);                                           \
        }                                                                    \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                \
        _simd_data_type2 operator()(const _simd_data_type2& src) const {     \
            auto vitem0 = _##_func_prefix##_abs_##_func_suffix1(src.val[0]); \
            auto vitem1 = _##_func_prefix##_abs_##_func_suffix1(src.val[1]); \
            return {{vitem0, vitem1}};                                       \
        }                                                                    \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                \
        _simd_data_type operator()(const _simd_data_type& src) const {       \
            return _##_func_prefix##_abs_##_func_suffix1(src);               \
        }                                                                    \
    };

OP(dt_int32, SIMDType::SSE4_2, "sse4.2", __m128i, __m128ix2, mm, epi32, si128,
   4)
OP(dt_int16, SIMDType::SSE4_2, "sse4.2", __m128i, __m128ix2, mm, epi16, si128,
   8)
OP(dt_int8, SIMDType::SSE4_2, "sse4.2", __m128i, __m128ix2, mm, epi8, si128, 16)

OP(dt_int32, SIMDType::AVX2, "avx2", __m256i, __m256ix2, mm256, epi32, si256, 8)
OP(dt_int16, SIMDType::AVX2, "avx2", __m256i, __m256ix2, mm256, epi16, si256,
   16)
OP(dt_int8, SIMDType::AVX2, "avx2", __m256i, __m256ix2, mm256, epi8, si256, 32)
#undef OP

#define OP(_ctype, _simd_type, _simd_target, _simd_data_type,               \
           _simd_data_type2, _func_prefix, _func_suffix, _simd_width)       \
    template <>                                                             \
    struct AbsOp<_simd_type, _ctype> : AbsOpBase<_simd_type, _ctype> {      \
        using AbsOpBase::AbsOpBase;                                         \
        using AbsOpBase::operator();                                        \
        constexpr static size_t SIMD_WIDTH = _simd_width;                   \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        void operator()(const _simd_data_type2& src, _ctype* dst) const {   \
            auto vitem = operator()(src);                                   \
            _##_func_prefix##_storeu_##_func_suffix(dst, vitem.val[0]);     \
            _##_func_prefix##_storeu_##_func_suffix(dst + SIMD_WIDTH,       \
                                                    vitem.val[1]);          \
        }                                                                   \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        _simd_data_type2 operator()(const _simd_data_type2& src) const {    \
            _simd_data_type zero_val =                                      \
                    _##_func_prefix##_set1_##_func_suffix(0.f);             \
            auto val0 = _##_func_prefix##_sub_##_func_suffix(zero_val,      \
                                                             src.val[0]);   \
            auto val1 = _##_func_prefix##_sub_##_func_suffix(zero_val,      \
                                                             src.val[1]);   \
            auto vitem0 =                                                   \
                    _##_func_prefix##_max_##_func_suffix(src.val[0], val0); \
            auto vitem1 =                                                   \
                    _##_func_prefix##_max_##_func_suffix(src.val[1], val1); \
            return {{vitem0, vitem1}};                                      \
        }                                                                   \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                               \
        _simd_data_type operator()(const _simd_data_type& src) const {      \
            _simd_data_type zero_val =                                      \
                    _##_func_prefix##_set1_##_func_suffix(0.f);             \
            auto val = _##_func_prefix##_sub_##_func_suffix(zero_val, src); \
            return _##_func_prefix##_max_##_func_suffix(src, val);          \
        }                                                                   \
    };

OP(dt_float32, SIMDType::SSE4_2, "sse4.2", __m128, __m128x2, mm, ps, 4)
OP(dt_float32, SIMDType::AVX2, "avx2", __m256, __m256x2, mm256, ps, 8)
#undef OP
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
