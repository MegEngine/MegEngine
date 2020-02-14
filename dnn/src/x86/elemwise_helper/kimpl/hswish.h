/**
 * \file dnn/src/x86/elemwise_helper/kimpl/hswish.h
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

template <SIMDType simd_type,typename src_ctype, typename dst_ctype = src_ctype>
struct HSwishOpBase : UnaryOpBase<simd_type,src_ctype, dst_ctype> {
    using UnaryOpBase<simd_type, src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmp = src;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        return (tmp);
    }
};

//! h_swish(x) = x * clip(x + 3, 0, 6) / 6
template <SIMDType simd_type, typename src_ctype, typename dst_type = src_ctype>
struct HSwishOp;

#define OP(_ctype, _simd_type, _simd_target, _simd_data_type,                  \
           _simd_data_type2, _func_prefix, _func_suffix, _simd_width)          \
    template <>                                                                \
    struct HSwishOp<_simd_type, _ctype> : HSwishOpBase<_simd_type, _ctype> {   \
        using HSwishOpBase::HSwishOpBase;                                      \
        using HSwishOpBase::operator();                                        \
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
            _simd_data_type val_0 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(0.0f);               \
            _simd_data_type val_6 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(6.0f);               \
            _simd_data_type val_3 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(3.0f);               \
            _simd_data_type val_1_6 =                                          \
                    _##_func_prefix##_set1_##_func_suffix(1.0f / 6);           \
            auto vitem0 =                                                      \
                    _##_func_prefix##_add_##_func_suffix(src.val[0], val_3);   \
            auto vitem1 =                                                      \
                    _##_func_prefix##_add_##_func_suffix(src.val[1], val_3);   \
            vitem0 = _##_func_prefix##_min_##_func_suffix(vitem0, val_6);      \
            vitem1 = _##_func_prefix##_min_##_func_suffix(vitem1, val_6);      \
            vitem0 = _##_func_prefix##_max_##_func_suffix(vitem0, val_0);      \
            vitem1 = _##_func_prefix##_max_##_func_suffix(vitem1, val_0);      \
            vitem0 = _##_func_prefix##_mul_##_func_suffix(vitem0, src.val[0]); \
            vitem1 = _##_func_prefix##_mul_##_func_suffix(vitem1, src.val[1]); \
            vitem0 = _##_func_prefix##_mul_##_func_suffix(vitem0, val_1_6);    \
            vitem1 = _##_func_prefix##_mul_##_func_suffix(vitem1, val_1_6);    \
            return {{vitem0, vitem1}};                                         \
        }                                                                      \
        MEGDNN_ATTRIBUTE_TARGET(_simd_target)                                  \
        _simd_data_type operator()(const _simd_data_type& src) const {         \
            _simd_data_type val_0 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(0.0f);               \
            _simd_data_type val_6 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(6.0f);               \
            _simd_data_type val_3 =                                            \
                    _##_func_prefix##_set1_##_func_suffix(3.0f);               \
            _simd_data_type val_1_6 =                                          \
                    _##_func_prefix##_set1_##_func_suffix(1.0f / 6);           \
            auto vitem = _##_func_prefix##_add_##_func_suffix(src, val_3);     \
            vitem = _##_func_prefix##_min_##_func_suffix(vitem, val_6);        \
            vitem = _##_func_prefix##_max_##_func_suffix(vitem, val_0);        \
            vitem = _##_func_prefix##_mul_##_func_suffix(src, vitem);          \
            return _##_func_prefix##_mul_##_func_suffix(vitem, val_1_6);       \
        }                                                                      \
    };

OP(dt_float32, SIMDType::SSE4_2, "sse4.2", __m128, __m128x2, mm, ps, 4)
OP(dt_float32, SIMDType::AVX2, "avx2", __m256, __m256x2, mm256, ps, 8)
#undef OP
#define OP(_ctype, _simd_type)                                                 \
    template <>                                                                \
    struct HSwishOp<_simd_type, _ctype> : HSwishOpBase<_simd_type, _ctype> {   \
        using HSwishOpBase::HSwishOpBase;                                      \
        using HSwishOpBase::operator();                                        \
    };
OP(dt_float32, SIMDType::NONE)
#undef OP

}  // namespace x86
}  // namespace megdnn
// vim: syntax=cpp.doxygen
