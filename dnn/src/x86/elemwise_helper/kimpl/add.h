/**
 * \file dnn/src/x86/elemwise_helper/kimpl/add.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/elemwise_helper/kimpl/op_binary_base.h"

namespace megdnn {
namespace x86 {

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct AddOpBase : BinaryOpBase<simd_type, src_ctype, dst_ctype> {
    using BinaryOpBase<simd_type, src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }

    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return src0 + src1;
    }
};

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct AddOp;

#define OP(_ctype, _simd_type, _simd_target, _simd_data_type,                 \
           _simd_data_type2, _ptr_type, _func_prefix, _func_suffix1,          \
           _func_suffix2, _simd_width)                                        \
    template <>                                                               \
    struct AddOp<_simd_type, _ctype> : AddOpBase<_simd_type, _ctype> {        \
        using AddOpBase::AddOpBase;                                           \
        using AddOpBase::operator();                                          \
        constexpr static size_t SIMD_WIDTH = _simd_width;                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                 \
        void operator()(const _simd_data_type2& src0,                         \
                        const _simd_data_type2& src1, _ctype* dst) const {    \
            auto vitem = operator()(src0, src1);                              \
            _##_func_prefix##_storeu_##_func_suffix2(                         \
                    reinterpret_cast<_ptr_type*>(dst), vitem.val[0]);         \
            _##_func_prefix##_storeu_##_func_suffix2(                         \
                    reinterpret_cast<_ptr_type*>(dst + SIMD_WIDTH),           \
                    vitem.val[1]);                                            \
        }                                                                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                 \
        _simd_data_type2 operator()(const _simd_data_type2& src0,             \
                                    const _simd_data_type2& src1) const {     \
            auto vitem0 = _##_func_prefix##_add_##_func_suffix1(src0.val[0],  \
                                                                src1.val[0]); \
            auto vitem1 = _##_func_prefix##_add_##_func_suffix1(src0.val[1],  \
                                                                src1.val[1]); \
            return {{vitem0, vitem1}};                                        \
        }                                                                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                 \
        _simd_data_type operator()(const _simd_data_type& src0,               \
                                   const _simd_data_type& src1) const {       \
            return _##_func_prefix##_add_##_func_suffix1(src0, src1);         \
        }                                                                     \
    };

OP(dt_float32, SIMDType::SSE4_2, "sse4.2", __m128, __m128x2, float, mm, ps, ps,
   4)
OP(dt_int32, SIMDType::SSE4_2, "sse4.2", __m128i, __m128ix2, __m128i, mm, epi32,
   si128, 4)
OP(dt_int16, SIMDType::SSE4_2, "sse4.2", __m128i, __m128ix2, __m128i, mm, epi16,
   si128, 8)
OP(dt_int8, SIMDType::SSE4_2, "sse4.2", __m128i, __m128ix2, __m128i, mm, epi8,
   si128, 16)

OP(dt_float32, SIMDType::AVX2, "avx2", __m256, __m256x2, float, mm256, ps, ps,
   8)
OP(dt_int32, SIMDType::AVX2, "avx2", __m256i, __m256ix2, __m256i, mm256, epi32,
   si256, 8)
OP(dt_int16, SIMDType::AVX2, "avx2", __m256i, __m256ix2, __m256i, mm256, epi16,
   si256, 16)
OP(dt_int8, SIMDType::AVX2, "avx2", __m256i, __m256ix2, __m256i, mm256, epi8,
   si256, 32)
#undef OP
#define OP(_ctype, _simd_type)                                         \
    template <>                                                        \
    struct AddOp<_simd_type, _ctype> : AddOpBase<_simd_type, _ctype> { \
        using AddOpBase::AddOpBase;                                    \
        using AddOpBase::operator();                                   \
    };

OP(dt_int32, SIMDType::NONE);
OP(dt_int16, SIMDType::NONE);
OP(dt_float32, SIMDType::NONE);
#undef OP
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
