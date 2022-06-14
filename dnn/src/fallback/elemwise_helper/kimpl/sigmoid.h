/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/sigmoid.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct SigmoidOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmpf = src;
        tmpf = exp(-tmpf);
        tmpf = 1.f / (1.f + tmpf);
        return tmpf;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct SigmoidOp;

#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width)              \
    template <>                                                                     \
    struct SigmoidOp<_ctype> : SigmoidOpBase<_ctype> {                              \
        using SigmoidOpBase::SigmoidOpBase;                                         \
        using SigmoidOpBase::operator();                                            \
        constexpr static size_t SIMD_WIDTH = _simd_width;                           \
        void operator()(const _simd_type2& src, _ctype* dst) const {                \
            auto vitem = operator()(src);                                           \
            GiStore##_func_suffix(dst, GiGetSubVector##_func_suffix##V2(vitem, 0)); \
            GiStore##_func_suffix(                                                  \
                    dst + SIMD_WIDTH, GiGetSubVector##_func_suffix##V2(vitem, 1));  \
        }                                                                           \
        void operator()(const _simd_type& src, _ctype* dst) const {                 \
            auto vitem = operator()(src);                                           \
            GiStore##_func_suffix(dst, vitem);                                      \
        }                                                                           \
        _simd_type2 operator()(const _simd_type2& src) const {                      \
            _simd_type2 ret;                                                        \
            GiSetSubVector##_func_suffix##V2(                                       \
                    ret, 0, operator()(GiGetSubVector##_func_suffix##V2(src, 0)));  \
            GiSetSubVector##_func_suffix##V2(                                       \
                    ret, 1, operator()(GiGetSubVector##_func_suffix##V2(src, 1)));  \
            return ret;                                                             \
        }                                                                           \
        _simd_type operator()(const _simd_type& src) const {                        \
            return GiSigmoidPs##_func_suffix(src);                                  \
        }                                                                           \
    };
OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
