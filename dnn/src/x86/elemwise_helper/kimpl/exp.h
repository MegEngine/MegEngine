/**
 * \file dnn/src/x86/elemwise_helper/kimpl/exp.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/elemwise/avx_util/avx_mathfun.h"
#include "src/x86/elemwise/sse_util/sse_mathfun.h"
#include "src/x86/elemwise_helper/kimpl/op_unary_base.h"
#include "src/x86/utils.h"

namespace megdnn {
namespace x86 {

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct ExpOpBase : UnaryOpBase<simd_type, src_ctype, dst_ctype> {
    using UnaryOpBase<simd_type, src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const { return exp(src); }
};

template <SIMDType simd_type, typename src_ctype, typename dst_type = src_ctype>
struct ExpOp;

#define OP(_ctype, _simd_type, _simd_target, _simd_data_type,             \
           _simd_data_type2, _func_prefix, _func_suffix, _simd_width,     \
           _func_name)                                                    \
    template <>                                                           \
    struct ExpOp<_simd_type, _ctype> : ExpOpBase<_simd_type, _ctype> {    \
        using ExpOpBase::ExpOpBase;                                       \
        using ExpOpBase::operator();                                      \
        constexpr static size_t SIMD_WIDTH = _simd_width;                 \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                             \
        void operator()(const _simd_data_type2& src, _ctype* dst) const { \
            auto vitem = operator()(src);                                 \
            _##_func_prefix##_storeu_##_func_suffix(dst, vitem.val[0]);   \
            _##_func_prefix##_storeu_##_func_suffix(dst + SIMD_WIDTH,     \
                                                    vitem.val[1]);        \
        }                                                                 \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                             \
        _simd_data_type2 operator()(const _simd_data_type2& src) const {  \
            auto vitem0 = _func_name##_##_func_suffix(src.val[0]);        \
            auto vitem1 = _func_name##_##_func_suffix(src.val[1]);        \
            return {{vitem0, vitem1}};                                    \
        }                                                                 \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                             \
        _simd_data_type operator()(const _simd_data_type& src) const {    \
            return _func_name##_##_func_suffix(src);                      \
        }                                                                 \
    };

OP(dt_float32, SIMDType::SSE4_2, "sse4.2", __m128, __m128x2, mm, ps, 4,
   detail::exp)
OP(dt_float32, SIMDType::AVX2, "avx2", __m256, __m256x2, mm256, ps, 8,
   detail::exp256)
#undef OP

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
