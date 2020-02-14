/**
 * \file dnn/src/x86/elemwise_helper/kimpl/fast_tanh.h
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

namespace megdnn {
namespace x86 {

//! tanh = x * (27 + x^2) / (27 + 9 * x^2)
template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct FastTanhOpBase : UnaryOpBase<simd_type, src_ctype, dst_ctype> {
    using UnaryOpBase< simd_type,src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float x = src;
        return x * (27.f + x * x) / (27.f + 9.f * x * x);
    }
};

template <SIMDType simd_type, typename src_ctype, typename dst_type = src_ctype>
struct FastTanhOp;

#define OP(_ctype, _simd_type, _simd_target, _simd_data_type,                  \
           _simd_data_type2, _func_prefix, _func_suffix, _simd_width)          \
    template <>                                                                \
    struct FastTanhOp<_simd_type, _ctype>                                      \
            : FastTanhOpBase<_simd_type, _ctype> {                             \
        using FastTanhOpBase::FastTanhOpBase;                                  \
        using FastTanhOpBase::operator();                                      \
        constexpr static size_t SIMD_WIDTH = _simd_width;                      \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        void operator()(const _simd_data_type2& src, _ctype* dst) const {      \
            auto vitem = operator()(src);                                      \
            _##_func_prefix##_storeu_##_func_suffix(dst, vitem.val[0]);        \
            _##_func_prefix##_storeu_##_func_suffix(dst + SIMD_WIDTH,          \
                                                    vitem.val[1]);             \
        }                                                                      \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        _simd_data_type2 operator()(const _simd_data_type2& src) const {       \
            _simd_data_type val_27 =                                           \
                    _##_func_prefix##_set1_##_func_suffix(27.0f);              \
            _simd_data_type val_9 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(9.0f);               \
            auto vitem0 = _##_func_prefix##_mul_##_func_suffix(src.val[0],     \
                                                               src.val[0]);    \
            auto vitem1 = _##_func_prefix##_mul_##_func_suffix(src.val[1],     \
                                                               src.val[1]);    \
            auto denominator0 =                                                \
                    _##_func_prefix##_mul_##_func_suffix(vitem0, val_9);       \
            auto denominator1 =                                                \
                    _##_func_prefix##_mul_##_func_suffix(vitem1, val_9);       \
            denominator0 = _##_func_prefix##_add_##_func_suffix(denominator0,  \
                                                                val_27);       \
            denominator1 = _##_func_prefix##_add_##_func_suffix(denominator1,  \
                                                                val_27);       \
            auto molecule0 =                                                   \
                    _##_func_prefix##_add_##_func_suffix(vitem0, val_27);      \
            auto molecule1 =                                                   \
                    _##_func_prefix##_add_##_func_suffix(vitem1, val_27);      \
            molecule0 = _##_func_prefix##_mul_##_func_suffix(molecule0,        \
                                                             src.val[0]);      \
            molecule1 = _##_func_prefix##_mul_##_func_suffix(molecule1,        \
                                                             src.val[1]);      \
            auto result0 = _##_func_prefix##_div_##_func_suffix(molecule0,     \
                                                                denominator0); \
            auto result1 = _##_func_prefix##_div_##_func_suffix(molecule1,     \
                                                                denominator1); \
            return {{result0, result1}};                                       \
        }                                                                      \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        _simd_data_type operator()(const _simd_data_type& src) const {         \
            _simd_data_type val_27 =                                           \
                    _##_func_prefix##_set1_##_func_suffix(27.0f);              \
            _simd_data_type val_9 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(9.0f);               \
            auto vitem = _##_func_prefix##_mul_##_func_suffix(src, src);       \
            auto denominator =                                                 \
                    _##_func_prefix##_mul_##_func_suffix(vitem, val_9);        \
            denominator =                                                      \
                    _##_func_prefix##_add_##_func_suffix(denominator, val_27); \
            auto molecule =                                                    \
                    _##_func_prefix##_add_##_func_suffix(vitem, val_27);       \
            molecule = _##_func_prefix##_mul_##_func_suffix(molecule, src);    \
            return _##_func_prefix##_div_##_func_suffix(molecule,              \
                                                        denominator);          \
        }                                                                      \
    };

OP(dt_float32, SIMDType::SSE4_2, "sse4.2", __m128, __m128x2, mm, ps, 4)
OP(dt_float32, SIMDType::AVX2, "avx2", __m256, __m256x2, mm256, ps, 8)
#undef OP

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
