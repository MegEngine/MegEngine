/**
 * \file dnn/src/x86/elemwise_helper/kimpl/fuse_add_sigmoid.h
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
#include "src/x86/elemwise_helper/kimpl/op_binary_base.h"
#include "src/x86/utils.h"

namespace megdnn {
namespace x86 {

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct FuseAddSigmoidOpBase : BinaryOpBase<simd_type, src_ctype, dst_ctype> {
    using BinaryOpBase<simd_type, src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        float tmpf = src0 + src1;
        tmpf = exp(-tmpf);
        tmpf = 1.f / (1.f + tmpf);
        return tmpf;
    }
};

template <SIMDType simd_type, typename src_ctype,
          typename dst_ctype = src_ctype>
struct FuseAddSigmoidOp;

#define OP(_ctype, _simd_type, _simd_target, _simd_data_type,                 \
           _simd_data_type2, _func_prefix, _func_suffix, _simd_width,         \
           _func_name)                                                        \
    template <>                                                               \
    struct FuseAddSigmoidOp<_simd_type, _ctype>                               \
            : FuseAddSigmoidOpBase<_simd_type, _ctype> {                      \
        using FuseAddSigmoidOpBase::FuseAddSigmoidOpBase;                     \
        using FuseAddSigmoidOpBase::operator();                               \
        constexpr static size_t SIMD_WIDTH = _simd_width;                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                 \
        void operator()(const _simd_data_type2& src0,                         \
                        const _simd_data_type2& src1, _ctype* dst) const {    \
            auto vitem = operator()(src0, src1);                              \
            _##_func_prefix##_storeu_##_func_suffix(dst, vitem.val[0]);       \
            _##_func_prefix##_storeu_##_func_suffix(dst + SIMD_WIDTH,         \
                                                    vitem.val[1]);            \
        }                                                                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                 \
        _simd_data_type2 operator()(const _simd_data_type2& src0,             \
                                    const _simd_data_type2& src1) const {     \
            return {{operator()(src0.val[0], src1.val[0]),                    \
                     operator()(src0.val[1], src1.val[1])}};                  \
        }                                                                     \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                 \
        _simd_data_type operator()(const _simd_data_type& src0,               \
                                   const _simd_data_type& src1) const {       \
            _simd_data_type zero_val =                                        \
                    _##_func_prefix##_set1_##_func_suffix(0.f);               \
            _simd_data_type one_val =                                         \
                    _##_func_prefix##_set1_##_func_suffix(1.f);               \
            auto val = _##_func_prefix##_add_##_func_suffix(src0, src1);      \
            val = _##_func_prefix##_sub_##_func_suffix(zero_val, val);        \
            val = _func_name##_##_func_suffix(val);                           \
            auto recipe = _##_func_prefix##_add_##_func_suffix(one_val, val); \
            val = _##_func_prefix##_div_##_func_suffix(one_val, recipe);      \
            return val;                                                       \
        }                                                                     \
    };
OP(dt_float32, SIMDType::SSE4_2, "sse4.2", __m128, __m128x2, mm, ps, 4,
   detail::exp)
OP(dt_float32, SIMDType::AVX2, "avx2", __m256, __m256x2, mm256, ps, 8,
   detail::exp256)
#undef OP
#define OP(_ctype, _simd_type)                            \
    template <>                                           \
    struct FuseAddSigmoidOp<_simd_type, _ctype>           \
            : FuseAddSigmoidOpBase<_simd_type, _ctype> {  \
        using FuseAddSigmoidOpBase::FuseAddSigmoidOpBase; \
        using FuseAddSigmoidOpBase::operator();           \
    };

OP(dt_float32, SIMDType::NONE)
#undef OP

}  // namespace x86
}  // namespace megdnn// vim: syntax=cpp.doxygen
