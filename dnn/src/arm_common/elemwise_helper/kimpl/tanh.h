/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/tanh.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/arm_common/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct TanhOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmp = src;
        return tanh(tmp);
    }
};

template <typename src_ctype, typename dst_type = src_ctype>
struct TanhOp;

#define OP(_ctype, _neon_type, _func_suffix, _simd_width)                     \
    template <>                                                               \
    struct TanhOp<_ctype> : TanhOpBase<_ctype> {                              \
        using TanhOpBase::TanhOpBase;                                         \
        using TanhOpBase::operator();                                         \
        constexpr static size_t SIMD_WIDTH = _simd_width;                     \
        void operator()(const _neon_type& src, _ctype* dst) const {           \
            auto vitem = operator()(src);                                     \
            vst1q_##_func_suffix(dst, vitem.val[0]);                          \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);             \
        }                                                                     \
        _neon_type operator()(const _neon_type& src) const {                  \
            auto one_val = vdupq_n_##_func_suffix(1.f);                       \
            auto two_val = vdupq_n_##_func_suffix(2.f);                       \
            auto val1 = src.val[0];                                           \
            auto val2 = src.val[1];                                           \
            val1 = vmulq_##_func_suffix(two_val, val1);                       \
            val2 = vmulq_##_func_suffix(two_val, val2);                       \
            val1 = exp_ps_##_func_suffix(val1);                               \
            val2 = exp_ps_##_func_suffix(val2);                               \
            val1 = vaddq_##_func_suffix(one_val, val1);                       \
            val2 = vaddq_##_func_suffix(one_val, val2);                       \
            auto rval1 = vrecpeq_##_func_suffix(val1);                        \
            auto rval2 = vrecpeq_##_func_suffix(val2);                        \
            rval1 = vmulq_##_func_suffix(vrecpsq_##_func_suffix(val1, rval1), \
                                         rval1);                              \
            rval2 = vmulq_##_func_suffix(vrecpsq_##_func_suffix(val2, rval2), \
                                         rval2);                              \
            val1 = vmulq_##_func_suffix(two_val, rval1);                      \
            val2 = vmulq_##_func_suffix(two_val, rval2);                      \
            val1 = vsubq_##_func_suffix(one_val, val1);                       \
            val2 = vsubq_##_func_suffix(one_val, val2);                       \
            return {{val1, val2}};                                            \
        }                                                                     \
    };
OP(dt_float32, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8x2_t, f16, 8)
#endif
#undef OP

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
