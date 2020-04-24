/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/fuse_mul_add3.h
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
struct FuseMulAdd3OpBase : TernaryOpBase<src_ctype, dst_ctype> {
    using TernaryOpBase<src_ctype, dst_ctype>::TernaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    const src_ctype src2, dst_ctype* dst) const {
        *dst = operator()(src0, src1, src2);
    }

    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1,
                         const src_ctype& src2) const {
        return (src0 * src1) + src2;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseMulAdd3Op;

#define OP(_ctype, _neon_type, _func_suffix, _simd_width)                     \
    template <>                                                               \
    struct FuseMulAdd3Op<_ctype> : FuseMulAdd3OpBase<_ctype> {                \
        using FuseMulAdd3OpBase::FuseMulAdd3OpBase;                           \
        using FuseMulAdd3OpBase::operator();                                  \
        constexpr static size_t SIMD_WIDTH = _simd_width;                     \
        void operator()(const _neon_type& src0, const _neon_type& src1,       \
                        const _neon_type& src2, dst_ctype* dst) const {       \
            auto vitem = operator()(src0, src1, src2);                        \
            vst1q_##_func_suffix(dst, vitem.val[0]);                          \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);             \
        }                                                                     \
        _neon_type operator()(const _neon_type& src0, const _neon_type& src1, \
                              const _neon_type& src2) const {                 \
            auto vitem0 = vmlaq_##_func_suffix(src2.val[0], src0.val[0],      \
                                               src1.val[0]);                  \
            auto vitem1 = vmlaq_##_func_suffix(src2.val[1], src0.val[1],      \
                                               src1.val[1]);                  \
            return {{vitem0, vitem1}};                                        \
        }                                                                     \
    };
OP(dt_float32, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8x2_t, f16, 8)
#endif
OP(dt_int32, int32x4x2_t, s32, 4)
OP(dt_int16, int16x8x2_t, s16, 8)
OP(dt_int8, int8x16x2_t, s8, 16)
#undef OP

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
