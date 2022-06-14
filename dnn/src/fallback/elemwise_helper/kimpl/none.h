/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/none.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct NoneOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    dst_ctype operator()(const src_ctype& src) const { return src; }
};

template <typename src_ctype, typename dst_type = src_ctype>
struct NoneOp;
#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width)            \
    template <>                                                                   \
    struct NoneOp<_ctype> : NoneOpBase<_ctype> {                                  \
        NoneOp(){};                                                               \
        NoneOp(float, float){};                                                   \
        using NoneOpBase::NoneOpBase;                                             \
        using NoneOpBase::operator();                                             \
        constexpr static size_t SIMD_WIDTH = _simd_width;                         \
        _simd_type2 operator()(const _simd_type2& src) const { return src; }      \
        void operator()(const _simd_type2& src, _ctype* dst) const {              \
            GiStore##_func_suffix(dst, GiGetSubVector##_func_suffix##V2(src, 0)); \
            GiStore##_func_suffix(                                                \
                    dst + SIMD_WIDTH, GiGetSubVector##_func_suffix##V2(src, 1));  \
        }                                                                         \
        void operator()(const _simd_type& src, _ctype* dst) const {               \
            GiStore##_func_suffix(dst, src);                                      \
        }                                                                         \
        _simd_type operator()(const _simd_type& src) const { return src; }        \
    };

OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
OP(dt_int32, GI_INT32_t, GI_INT32_V2_t, Int32, GI_SIMD_LEN_BYTE / sizeof(int32_t))
OP(dt_int8, GI_INT8_t, GI_INT8_V2_t, Int8, GI_SIMD_LEN_BYTE / sizeof(int8_t))
#undef OP

template <>
struct NoneOpBase<dt_qint8, dt_qint8> : UnaryOpBase<dt_qint8, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint8& src, dt_qint8* dst) const { *dst = src; }
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
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int32_t);

    void operator()(const GI_INT32_V2_t& vsrc, dt_qint8* dst) const {
        GiStoreInt32(dst, GiGetSubVectorInt32V2(vsrc, 0));
        GiStoreInt32(dst + 16, GiGetSubVectorInt32V2(vsrc, 1));
    }
    void operator()(const GI_INT32_t& src, dt_qint8* dst) const {
        GiStoreInt32(dst, src);
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
