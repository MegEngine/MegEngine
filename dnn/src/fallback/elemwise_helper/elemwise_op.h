/**
 * \file dnn/src/fallback/elemwise_helper/elemwise_op.h
 */

#pragma once

#include "src/fallback/elemwise_helper/op_binary.h"
#include "src/fallback/elemwise_helper/op_common.h"
#include "src/fallback/elemwise_helper/op_ternary.h"
#include "src/fallback/elemwise_helper/op_unary.h"

#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/general_intrinsic/gi_int.h"

namespace megdnn {
namespace elemwise {

///////////////////////////////// ParamElemVistor ///////////////////////////

#define cb(_ctype, _inner_ctype, _simd_type, _fun_suffix)         \
    template <>                                                   \
    struct ParamElemVisitor<_ctype> {                             \
        _simd_type operator()(const _ctype* src) const {          \
            return GiLoad##_fun_suffix(src);                      \
        }                                                         \
    };                                                            \
    template <>                                                   \
    struct ParamElemVisitorDup<_ctype> {                          \
        _simd_type operator()(const _ctype* src) const {          \
            return GiBroadcast##_fun_suffix(                      \
                    *reinterpret_cast<const _inner_ctype*>(src)); \
        }                                                         \
    }
cb(dt_qint32, int32_t, GI_INT32_t, Int32);
cb(dt_qint8, int8_t, GI_INT8_t, Int8);

cb(dt_float32, float, GI_FLOAT32_t, Float32);
cb(dt_int32, int32_t, GI_INT32_t, Int32);
cb(dt_int8, int8_t, GI_INT8_t, Int8);
#undef cb

template <typename ctype>
struct ParamElemVisitorBcast101x4;
#define cb(_ctype, _inner_ctype, _simd_type, _fun_suffix, rel_suffix)              \
    template <>                                                                    \
    struct ParamElemVisitorBcast101x4<_ctype> {                                    \
        _simd_type operator()(const _ctype* src) const {                           \
            return GiReinter##rel_suffix##To##_fun_suffix(GiBroadcast##rel_suffix( \
                    *reinterpret_cast<const _inner_ctype*>(src)));                 \
        }                                                                          \
    }

cb(dt_qint8, int32_t, GI_INT8_t, Int8, Int32);
cb(dt_int8, int32_t, GI_INT8_t, Int8, Int32);
#undef cb
#define cb(_ctype, _inner_ctype, _simd_type, _fun_suffix) \
    template <>                                           \
    struct ParamElemVisitorBcast101x4<_ctype> {           \
        _simd_type operator()(const _ctype* src) const {  \
            return GiLoad##_fun_suffix(src);              \
        }                                                 \
    }

cb(dt_qint32, int32_t, GI_INT32_t, Int32);
cb(dt_float32, float, GI_FLOAT32_t, Float32);
cb(dt_int32, int32_t, GI_INT32_t, Int32);
#undef cb

}  // namespace elemwise
}  // namespace megdnn

// vim: syntax=cpp.doxygen
