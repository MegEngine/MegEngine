/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/rmulh.h
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
struct RmulhOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return round_mulh_saturate(src0, src1);
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct RmulhOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)        \
    template <>                                                               \
    struct RmulhOp<_ctype> : RmulhOpBase<_ctype> {                            \
        using RmulhOpBase::RmulhOpBase;                                       \
        using RmulhOpBase::operator();                                        \
        constexpr static size_t SIMD_WIDTH = _simd_width;                     \
        void operator()(const _neon_type2& src0, const _neon_type2& src1,     \
                        dst_ctype* dst) const {                               \
            auto vitem = operator()(src0, src1);                              \
            vst1q_##_func_suffix(dst, vitem.val[0]);                          \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);             \
        }                                                                     \
        _neon_type2 operator()(const _neon_type2& src0,                       \
                               const _neon_type2& src1) const {               \
            auto vitem0 = vqrdmulhq_##_func_suffix(src0.val[0], src1.val[0]); \
            auto vitem1 = vqrdmulhq_##_func_suffix(src0.val[1], src1.val[1]); \
            return {{vitem0, vitem1}};                                        \
        }                                                                     \
        void operator()(const _neon_type& src0, const _neon_type& src1,       \
                        dst_ctype* dst) const {                               \
            auto vitem = operator()(src0, src1);                              \
            vst1q_##_func_suffix(dst, vitem);                                 \
        }                                                                     \
        _neon_type operator()(const _neon_type& src0,                         \
                              const _neon_type& src1) const {                 \
            return vqrdmulhq_##_func_suffix(src0, src1);                      \
        }                                                                     \
    };
OP(dt_int32, int32x4_t, int32x4x2_t, s32, 4)
OP(dt_int16, int16x8_t, int16x8x2_t, s16, 8)
#undef OP
/**
 * As There is no vqrdmulh.s8, we have to emulate it manually as this is
 * requested by the researchers
 */
template <>
struct RmulhOp<dt_int8> : RmulhOpBase<dt_int8> {
    using RmulhOpBase::RmulhOpBase;
    using RmulhOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 16;
    void operator()(const int8x16x2_t& src0, const int8x16x2_t& src1,
                    int8_t* dst) const {
        auto vitem = operator()(src0, src1);
        vst1q_s8(dst, vitem.val[0]);
        vst1q_s8(dst + SIMD_WIDTH, vitem.val[1]);
    }
    int8x16x2_t operator()(const int8x16x2_t& src0,
                           const int8x16x2_t& src1) const {
        int8x16_t val, var;
        int8x8_t lol, hil, lor, hir;
        int16x8_t mu0, mu1;

        val = src0.val[0];
        var = src1.val[0];
        lol = vget_low_s8(val);
        hil = vget_high_s8(val);
        lor = vget_low_s8(var);
        hir = vget_high_s8(var);

        mu0 = vmull_s8(lol, lor);
        lol = vqrshrn_n_s16(mu0, 7);
        mu1 = vmull_s8(hil, hir);
        hil = vqrshrn_n_s16(mu1, 7);

        int8x16_t val1, var1;
        int8x8_t lol1, hil1, lor1, hir1;
        int16x8_t mu01, mu11;

        val1 = src0.val[1];
        var1 = src1.val[1];
        lol1 = vget_low_s8(val1);
        hil1 = vget_high_s8(val1);
        lor1 = vget_low_s8(var1);
        hir1 = vget_high_s8(var1);

        mu01 = vmull_s8(lol1, lor1);
        lol1 = vqrshrn_n_s16(mu01, 7);
        mu11 = vmull_s8(hil1, hir1);
        hil1 = vqrshrn_n_s16(mu11, 7);

        return {{vcombine_s8(lol, hil), vcombine_s8(lol1, hil1)}};
    }
    void operator()(const int8x16_t& src0, const int8x16_t& src1,
                    int8_t* dst) const {
        auto vitem = operator()(src0, src1);
        vst1q_s8(dst, vitem);
    }
    int8x16_t operator()(const int8x16_t& src0, const int8x16_t& src1) const {
        int8x16_t val, var;
        int8x8_t lol, hil, lor, hir;
        int16x8_t mu0, mu1;

        val = src0;
        var = src1;
        lol = vget_low_s8(val);
        hil = vget_high_s8(val);
        lor = vget_low_s8(var);
        hir = vget_high_s8(var);

        mu0 = vmull_s8(lol, lor);
        lol = vqrshrn_n_s16(mu0, 7);
        mu1 = vmull_s8(hil, hir);
        hil = vqrshrn_n_s16(mu1, 7);

        return vcombine_s8(lol, hil);
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
