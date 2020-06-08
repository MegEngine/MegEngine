/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/none.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "src/arm_common/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct NoneOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    dst_ctype operator()(const src_ctype& src) const { return src; }
};

template <typename src_ctype, typename dst_type = src_ctype>
struct NoneOp;
#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)       \
    template <>                                                              \
    struct NoneOp<_ctype> : NoneOpBase<_ctype> {                             \
        NoneOp(){};                                                          \
        NoneOp(float, float){};                                              \
        using NoneOpBase::NoneOpBase;                                        \
        using NoneOpBase::operator();                                        \
        constexpr static size_t SIMD_WIDTH = _simd_width;                    \
        _neon_type2 operator()(const _neon_type2& src) const { return src; } \
        void operator()(const _neon_type2& src, _ctype* dst) const {         \
            vst1q_##_func_suffix(dst, src.val[0]);                           \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, src.val[1]);              \
        }                                                                    \
        void operator()(const _neon_type& src, _ctype* dst) const {          \
            vst1q_##_func_suffix(dst, src);                                  \
        }                                                                    \
        _neon_type operator()(const _neon_type& src) const { return src; }   \
    };

OP(dt_float32, float32x4_t, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8_t, float16x8x2_t, f16, 8)
#endif
OP(dt_int32, int32x4_t, int32x4x2_t, s32, 4)
OP(dt_int16, int16x8_t, int16x8x2_t, s16, 8)
OP(dt_int8, int8x16_t, int8x16x2_t, s8, 16)
#undef OP

template <>
struct NoneOpBase<dt_qint8, dt_qint8> : UnaryOpBase<dt_qint8, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint8& src, dt_qint8* dst) const { *dst = src; }
};

template <>
struct NoneOpBase<dt_quint8, dt_quint8> : UnaryOpBase<dt_quint8, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_quint8& src, dt_quint8* dst) const { *dst = src; }
};

template <>
struct NoneOpBase<dt_qint32, dt_qint8> : UnaryOpBase<dt_qint32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint32& src, dt_qint8* dst) const {
        *(reinterpret_cast<dt_qint32*>(dst)) = src;
    }
};

template <>
struct NoneOp<dt_qint32, dt_qint8> : NoneOpBase<dt_qint32, dt_qint8> {
    using NoneOpBase::NoneOpBase;
    using NoneOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc, dt_qint8* dst) const {
        vst1q_s32(reinterpret_cast<int32_t*>(dst), vsrc.val[0]);
        vst1q_s32(reinterpret_cast<int32_t*>(dst + 16), vsrc.val[1]);
    }
    void operator()(const int32x4_t& src, dt_qint8* dst) const {
        vst1q_s32(reinterpret_cast<int32_t*>(dst), src);
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
