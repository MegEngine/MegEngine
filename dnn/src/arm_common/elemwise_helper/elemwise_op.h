#pragma once

#include "src/arm_common/elemwise_helper/op_binary.h"
#include "src/arm_common/elemwise_helper/op_ternary.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/fallback/elemwise_helper/op_common.h"

namespace megdnn {
namespace elemwise {

using BcastType = megdnn::elemwise::BcastType;

///////////////////////////////// ParamElemVistor ///////////////////////////

#define cb(_ctype, _inner_ctype, _neon_type, _fun_suffix, _neon_type_v2)               \
    template <>                                                                        \
    struct ParamElemVisitor<_ctype> {                                                  \
        _neon_type operator()(const _ctype* src) const {                               \
            return vld1q_##_fun_suffix(reinterpret_cast<const _inner_ctype*>(src));    \
        }                                                                              \
    };                                                                                 \
    template <>                                                                        \
    struct ParamElemVisitorDup<_ctype> {                                               \
        _neon_type operator()(const _ctype* src) const {                               \
            return vdupq_n_##_fun_suffix(*reinterpret_cast<const _inner_ctype*>(src)); \
        }                                                                              \
    };                                                                                 \
    template <>                                                                        \
    struct ParamElemVisitorV2<_ctype> {                                                \
        _neon_type_v2 operator()(const _ctype* src, const _ctype* src_1) const {       \
            _neon_type_v2 ret;                                                         \
            ret.val[0] =                                                               \
                    vld1q_##_fun_suffix(reinterpret_cast<const _inner_ctype*>(src));   \
            ret.val[1] =                                                               \
                    vld1q_##_fun_suffix(reinterpret_cast<const _inner_ctype*>(src_1)); \
            return ret;                                                                \
        }                                                                              \
    };                                                                                 \
    template <>                                                                        \
    struct ParamElemVisitorDupV2<_ctype> {                                             \
        _neon_type_v2 operator()(const _ctype* src) const {                            \
            _neon_type_v2 ret;                                                         \
            ret.val[0] = vdupq_n_##_fun_suffix(                                        \
                    *reinterpret_cast<const _inner_ctype*>(src));                      \
            ret.val[1] = ret.val[0];                                                   \
            return ret;                                                                \
        }                                                                              \
    }
cb(dt_quint8, uint8_t, uint8x16_t, u8, uint8x16x2_t);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
cb(__fp16, __fp16, float16x8_t, f16, float16x8x2_t);
#endif
cb(dt_int16, int16_t, int16x8_t, s16, int16x8x2_t);
#undef cb

template <typename ctype>
struct ParamElemVisitorBcast101x4;
#define cb(_ctype, _inner_ctype, _neon_type, _fun_suffix, rel_suffix, _neon_type_v2)   \
    template <>                                                                        \
    struct ParamElemVisitorBcast101x4<_ctype> {                                        \
        _neon_type operator()(const _ctype* src) const {                               \
            return vreinterpretq_##_fun_suffix##_##rel_suffix(vld1q_dup_##rel_suffix(  \
                    reinterpret_cast<const _inner_ctype*>(src)));                      \
        }                                                                              \
    };                                                                                 \
    template <>                                                                        \
    struct ParamElemVisitorBcast101x4V2<_ctype> {                                      \
        _neon_type_v2 operator()(const _ctype* src) const {                            \
            _neon_type_v2 ret;                                                         \
            ret.val[0] =                                                               \
                    vreinterpretq_##_fun_suffix##_##rel_suffix(vld1q_dup_##rel_suffix( \
                            reinterpret_cast<const _inner_ctype*>(src)));              \
            ret.val[1] = ret.val[0];                                                   \
            return ret;                                                                \
        }                                                                              \
    }

cb(dt_quint8, uint32_t, uint8x16_t, u8, u32, uint8x16x2_t);
cb(dt_int16, int64_t, int16x8_t, s16, s64, int16x8x2_t);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
cb(__fp16, uint64_t, float16x8_t, f16, u64, float16x8x2_t);
#endif
#undef cb

template <typename ctype>
struct ParamElemVisitorBcast101x8;
#define cb(_ctype, _inner_ctype, _neon_type, _fun_suffix)                           \
    template <>                                                                     \
    struct ParamElemVisitorBcast101x8<_ctype> {                                     \
        _neon_type operator()(const _ctype* src) const {                            \
            return vld1q_##_fun_suffix(reinterpret_cast<const _inner_ctype*>(src)); \
        }                                                                           \
    }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
cb(__fp16, __fp16, float16x8_t, f16);
#endif
#undef cb

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
struct OpCallerBinaryBcast101xXVec<__fp16, 8> {
    using src_ctype = __fp16;

    template <typename Op>
    static void run(
            const src_ctype* src0, const src_ctype* src1, typename Op::dst_ctype* dst,
            const Op& op, size_t batch, size_t nr_channel_blocks,
            size_t channel_stride) {
        ParamElemVisitorBcast101x8<src_ctype> vis0;
        ParamElemVisitor<src_ctype> vis1;
        OpCallerBinaryBcast101xDVec<src_ctype, 8>::run(
                src0, src1, dst, op, vis0, vis1, batch, nr_channel_blocks,
                channel_stride);
    }
};

template <>
struct OpCallerBinaryVecBcast101xX<__fp16, 8> {
    using src_ctype = __fp16;
    template <typename Op>
    static void run(
            const src_ctype* src0, const src_ctype* src1, typename Op::dst_ctype* dst,
            const Op& op, size_t batch, size_t nr_channel_blocks,
            size_t channel_stride) {
        ParamElemVisitor<src_ctype> vis0;
        ParamElemVisitorBcast101x8<src_ctype> vis1;
        OpCallerBinaryVecBcast101xD<src_ctype, 8>::run(
                src0, src1, dst, op, vis0, vis1, batch, nr_channel_blocks,
                channel_stride);
    }
};

template <>
struct OpCallerTernaryBcast101xXVecBcast101xX<__fp16, 8> {
    using src_ctype = __fp16;
    template <typename Op>
    static void run(
            const src_ctype* src0, const src_ctype* src1, const src_ctype* src2,
            typename Op::dst_ctype* dst, const Op& op, size_t batch,
            size_t nr_channel_blocks, size_t channel_stride) {
        ParamElemVisitorBcast101x8<src_ctype> vis0;
        ParamElemVisitor<src_ctype> vis1;
        ParamElemVisitorBcast101x8<src_ctype> vis2;
        OpCallerTernaryBcast101xDVecBcast101xD<src_ctype, 8>::run(
                src0, src1, src2, dst, op, vis0, vis1, vis2, batch, nr_channel_blocks,
                channel_stride);
    }
};

template <>
struct OpCallerTernaryVecBcast101xXVec<__fp16, 8> {
    using src_ctype = __fp16;
    template <typename Op>
    static void run(
            const src_ctype* src0, const src_ctype* src1, const src_ctype* src2,
            typename Op::dst_ctype* dst, const Op& op, size_t batch,
            size_t nr_channel_blocks, size_t channel_stride) {
        ParamElemVisitor<src_ctype> vis0;
        ParamElemVisitorBcast101x8<src_ctype> vis1;
        ParamElemVisitor<src_ctype> vis2;
        OpCallerTernaryVecBcast101xDVec<src_ctype, 8>::run(
                src0, src1, src2, dst, op, vis0, vis1, vis2, batch, nr_channel_blocks,
                channel_stride);
    }
};
#endif

}  // namespace elemwise
}  // namespace megdnn

// vim: syntax=cpp.doxygen
