/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/exp.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct ExpOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmp = src;
        return exp(tmp);
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct ExpOp;

#define OP(_ctype, _simd_type, _func_suffix, _simd_width)                            \
    template <>                                                                      \
    struct ExpOp<_ctype> : ExpOpBase<_ctype> {                                       \
        using ExpOpBase::ExpOpBase;                                                  \
        using ExpOpBase::operator();                                                 \
        constexpr static size_t SIMD_WIDTH = _simd_width;                            \
        void operator()(const _simd_type& src, _ctype* dst) const {                  \
            auto vitem = operator()(src);                                            \
            GiStore##_func_suffix(dst, GiGetSubVector##_func_suffix##V2(vitem, 0));  \
            GiStore##_func_suffix(                                                   \
                    dst + SIMD_WIDTH, GiGetSubVector##_func_suffix##V2(vitem, 1));   \
        }                                                                            \
        _simd_type operator()(const _simd_type& src) const {                         \
            auto vitem0 =                                                            \
                    GiExpPs##_func_suffix(GiGetSubVector##_func_suffix##V2(src, 0)); \
            auto vitem1 =                                                            \
                    GiExpPs##_func_suffix(GiGetSubVector##_func_suffix##V2(src, 1)); \
            _simd_type ret;                                                          \
            GiSetSubVector##_func_suffix##V2(ret, 0, vitem0);                        \
            GiSetSubVector##_func_suffix##V2(ret, 1, vitem1);                        \
            return ret;                                                              \
        }                                                                            \
    };
OP(dt_float32, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
