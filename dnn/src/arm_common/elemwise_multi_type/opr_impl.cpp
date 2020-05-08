/**
 * \file dnn/src/arm_common/elemwise_multi_type/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./opr_impl.h"
#include "src/common/elemwise_multi_type/kern_defs.cuh"
#include "src/naive/handle.h"

#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace {

using namespace megdnn;

template <int k>
void neon_round_shr_saturate_int16_static_k(const int16_t* a_ptr, size_t size,
                                            int8_t* dst_ptr) {
    static_assert(k >= 1 && k <= 8, "Shift offset out of range");
    size_t i = 0;
    int16x8_t x0, x1, f0, f1;
    for (; i + 15 < size; i += 16, a_ptr += 16, dst_ptr += 16) {
        x0 = vld1q_s16(a_ptr);
        x1 = vld1q_s16(a_ptr + 8);
        f0 = vshrq_n_s16(x0, 15);
        f1 = vshrq_n_s16(x1, 15);
        x0 = vqaddq_s16(x0, f0);
        x1 = vqaddq_s16(x1, f1);
        vst1_s8(dst_ptr, vqrshrn_n_s16(x0, k));
        vst1_s8(dst_ptr + 8, vqrshrn_n_s16(x1, k));
    }
    for (; i < size; i++, a_ptr++, dst_ptr++) {
        *dst_ptr = megdnn::elemwise_multi_type::round_shr_saturate<int16_t,
                                                                   int8_t>(
                *a_ptr, k);
    }
}

}  // namespace

namespace megdnn {
namespace arm_common {

template <typename stype>
void ElemwiseMultiTypeImpl::neon_round_shr_saturate_bcast_scalar(
        const stype* a_ptr, int8_t k, size_t size, dt_int8* dst_ptr) {
    MEGDNN_MARK_USED_VAR(a_ptr);
    MEGDNN_MARK_USED_VAR(k);
    MEGDNN_MARK_USED_VAR(size);
    MEGDNN_MARK_USED_VAR(dst_ptr);
    megdnn_throw(
            "ElemwiseMultiType (mode=ROUND_SHR_SATURATE) only supports int8, "
            "int16 and int32 on ARM");
}

template <>
void ElemwiseMultiTypeImpl::neon_round_shr_saturate_bcast_scalar<int8_t>(
        const int8_t* a_ptr, int8_t k, size_t size, dt_int8* dst_ptr) {
    size_t i = 0;
    const int8x16_t shift_vec = vdupq_n_s8(-k);
    int8x16_t x0, x1, f0, f1;
    for (; i + 31 < size; i += 32, a_ptr += 32, dst_ptr += 32) {
        x0 = vld1q_s8(a_ptr);
        x1 = vld1q_s8(a_ptr + 16);
        f0 = vshrq_n_s8(x0, 7);
        f1 = vshrq_n_s8(x1, 7);
        x0 = vqaddq_s8(x0, f0);
        x1 = vqaddq_s8(x1, f1);
        vst1q_s8(dst_ptr, vrshlq_s8(x0, shift_vec));
        vst1q_s8(dst_ptr + 16, vrshlq_s8(x1, shift_vec));
    }
    for (; i < size; i++, a_ptr++, dst_ptr++) {
        *dst_ptr = elemwise_multi_type::round_shr_saturate<int8_t, int8_t>(
                *a_ptr, k);
    }
}

template <>
void ElemwiseMultiTypeImpl::neon_round_shr_saturate_bcast_scalar<int16_t>(
        const int16_t* a_ptr, int8_t k, size_t size, dt_int8* dst_ptr) {
    // vqrshrn_n_s16 is significantly faster than vrshlq_s16 + vqmovn_s16, but
    // it requires that shift offset is known at compile time.
    switch (k) {
#define DISPATCH(i)                                                      \
    case i:                                                              \
        neon_round_shr_saturate_int16_static_k<i>(a_ptr, size, dst_ptr); \
        return;
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(3)
        DISPATCH(4)
        DISPATCH(5)
        DISPATCH(6)
        DISPATCH(7)
        DISPATCH(8)
#undef DISPATCH
        default:
            break;
    }

    size_t i = 0;
    const int16x8_t shift_vec = vdupq_n_s16(-k);
    int16x8_t x0, x1, f0, f1;
    for (; i + 15 < size; i += 16, a_ptr += 16, dst_ptr += 16) {
        x0 = vld1q_s16(a_ptr);
        x1 = vld1q_s16(a_ptr + 8);
        f0 = vshrq_n_s16(x0, 15);
        f1 = vshrq_n_s16(x1, 15);
        x0 = vqaddq_s16(x0, f0);
        x1 = vqaddq_s16(x1, f1);
        vst1_s8(dst_ptr, vqmovn_s16(vrshlq_s16(x0, shift_vec)));
        vst1_s8(dst_ptr + 8, vqmovn_s16(vrshlq_s16(x1, shift_vec)));
    }
    for (; i < size; i++, a_ptr++, dst_ptr++) {
        *dst_ptr = elemwise_multi_type::round_shr_saturate<int16_t, int8_t>(
                *a_ptr, k);
    }
}

template <>
void ElemwiseMultiTypeImpl::neon_round_shr_saturate_bcast_scalar<int32_t>(
        const int32_t* a_ptr, int8_t k, size_t size, dt_int8* dst_ptr) {
    size_t i = 0;
    const int32x4_t shift_vec = vdupq_n_s32(-k);
    int32x4_t x0, x1, f0, f1;
    int8x8_t o0;
    for (; i + 7 < size; i += 8, a_ptr += 8, dst_ptr += 8) {
        x0 = vld1q_s32(a_ptr);
        x1 = vld1q_s32(a_ptr + 4);
        f0 = vshrq_n_s32(x0, 31);
        f1 = vshrq_n_s32(x1, 31);
        x0 = vqaddq_s32(x0, f0);
        x1 = vqaddq_s32(x1, f1);
        o0 = vqmovn_s16(vcombine_s16(vqmovn_s32(vrshlq_s32(x0, shift_vec)),
                                     vqmovn_s32(vrshlq_s32(x1, shift_vec))));
        vst1_s8(dst_ptr, o0);
    }
    for (; i < size; i++, a_ptr++, dst_ptr++) {
        *dst_ptr = elemwise_multi_type::round_shr_saturate<int32_t, int8_t>(
                *a_ptr, k);
    }
}

template <typename ctype>
void ElemwiseMultiTypeImpl::dispatch_round_shr_saturate_iXxi8xi8_bcast_scalar(
        const ElemwiseOpParamN<2>& param, megdnn::dt_int8* dst) {
    auto a_ptr = param[0].ptr<ctype>();
    auto k = param[1].ptr<dt_int8>()[0];
    size_t size = param.size;

    MEGDNN_DISPATCH_CPU_KERN_OPR(
            neon_round_shr_saturate_bcast_scalar(a_ptr, k, size, dst));
}

void ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi8(
        const ElemwiseOpParamN<2>& param, megdnn::dt_int8* dst) {
    if (is_vector(param[0].layout) && is_broadcasted_scalar(param[1].layout)) {
        switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                     \
    case DTypeTrait<t>::enumv:                                    \
        return dispatch_round_shr_saturate_iXxi8xi8_bcast_scalar< \
                DTypeTrait<t>::ctype>(param, dst);
            MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
            default:
                megdnn_throw(
                        "ElemwiseMultiType (mode=ROUND_SHR_SATURATE) only "
                        "supports int8, int16 and int32 on ARM");
        }
    }

    fallback::ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi8(param, dst);
}

void neon_fuse_add_rmulh_round_shr_saturate_bcast_1c11_int16(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const int16_t* x_ptr, const int16_t* b_ptr, const int16_t M,
        const int offset, const int8_t minv, const int8_t maxv, size_t size,
        int8_t* dst_ptr) {
    MEGDNN_MARK_USED_VAR(size);
    const int16x8_t shift_vec = vdupq_n_s16(-offset);
    const int16x8_t M_vec = vdupq_n_s16(M);
    const int8x16_t minv_vec = vdupq_n_s8(minv);
    const int8x16_t maxv_vec = vdupq_n_s8(maxv);

    size_t i = 0, b_pos = 0, channel_offset = 0;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        b_pos = 0;
        for (size_t chan = 0; chan < channel_size; ++chan, ++b_pos) {
            auto b_vec = vdupq_n_s16(b_ptr[b_pos]);
            channel_offset += channel_stride;
            for (; i + 15 < channel_offset;
                 i += 16, x_ptr += 16, dst_ptr += 16) {
                auto x0 = vld1q_s16(x_ptr);
                auto x1 = vld1q_s16(x_ptr + 8);
                x0 = vaddq_s16(x0, b_vec);
                x1 = vaddq_s16(x1, b_vec);
                x0 = vqrdmulhq_s16(x0, M_vec);
                x1 = vqrdmulhq_s16(x1, M_vec);
                // FIXME Theoretically, we should check shift != 0 here,
                auto fixup0 = vshrq_n_s16(x0, 15);
                auto fixup1 = vshrq_n_s16(x1, 15);
                x0 = vqaddq_s16(x0, fixup0);
                x1 = vqaddq_s16(x1, fixup1);
                auto o0 = vcombine_s8(vqmovn_s16(vrshlq_s16(x0, shift_vec)),
                                      vqmovn_s16(vrshlq_s16(x1, shift_vec)));
                o0 = vminq_s8(o0, maxv_vec);
                o0 = vmaxq_s8(o0, minv_vec);
                vst1q_s8(dst_ptr, o0);
            }
            for (; i + 7 < channel_offset; i += 8, x_ptr += 8, dst_ptr += 8) {
                auto x0 = vld1q_s16(x_ptr);
                x0 = vaddq_s16(x0, b_vec);
                x0 = vqrdmulhq_s16(x0, M_vec);
                // FIXME Theoretically, we should check shift != 0 here,
                auto fixup0 = vshrq_n_s16(x0, 15);
                x0 = vqaddq_s16(x0, fixup0);
                auto o0 = vqmovn_s16(vrshlq_s16(x0, shift_vec));
                o0 = vmin_s8(o0, vget_low_s8(maxv_vec));
                o0 = vmax_s8(o0, vget_low_s8(minv_vec));
                vst1_s8(dst_ptr, o0);
            }
            dt_int16 bias = b_ptr[b_pos];
            for (; i < channel_offset; ++i, ++x_ptr, ++dst_ptr) {
                dt_int16 result = rounding_shift_right_away_from_zero(
                        round_mulh_saturate<dt_int16>(*x_ptr + bias, M),
                        offset);
                *dst_ptr = static_cast<dt_int8>(std::max<dt_int16>(
                        std::min<dt_int16>(result, maxv), minv));
            }
        }
    }
}

void neon_fuse_add_rmulh_round_shr_saturate_bcast_1c11_int32(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const int32_t* x_ptr, const int32_t* b_ptr, const int32_t M,
        const int offset, const int8_t minv, const int8_t maxv, size_t size,
        int8_t* dst_ptr) {
    MEGDNN_MARK_USED_VAR(size);
    const int32x4_t shift_vec = vdupq_n_s32(-offset);
    const int32x4_t M_vec = vdupq_n_s32(M);
    const int8x8_t minv_vec = vdup_n_s8(minv);
    const int8x8_t maxv_vec = vdup_n_s8(maxv);

    size_t i = 0, b_pos = 0, channel_offset = 0;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        b_pos = 0;
        for (size_t chan = 0; chan < channel_size; ++chan, ++b_pos) {
            int32x4_t b_vec = vdupq_n_s32(b_ptr[b_pos]);
            channel_offset += channel_stride;
            for (; i + 7 < channel_offset; i += 8, x_ptr += 8, dst_ptr += 8) {
                auto x0 = vld1q_s32(x_ptr);
                auto x1 = vld1q_s32(x_ptr + 4);
                x0 = vaddq_s32(x0, b_vec);
                x1 = vaddq_s32(x1, b_vec);
                x0 = vqrdmulhq_s32(x0, M_vec);
                x1 = vqrdmulhq_s32(x1, M_vec);
                // FIXME Theoretically, we should check shift != 0 here,
                auto fixup0 = vshrq_n_s32(x0, 31);
                auto fixup1 = vshrq_n_s32(x1, 31);
                x0 = vqaddq_s32(x0, fixup0);
                x1 = vqaddq_s32(x1, fixup1);
                auto o0 = vqmovn_s32(vrshlq_s32(x0, shift_vec));
                auto o1 = vqmovn_s32(vrshlq_s32(x1, shift_vec));
                auto of = vqmovn_s16(vcombine_s16(o0, o1));
                of = vmin_s8(of, maxv_vec);
                of = vmax_s8(of, minv_vec);
                vst1_s8(dst_ptr, of);
            }
            dt_int32 bias = b_ptr[b_pos];
            for (; i < channel_offset; ++i, ++x_ptr, ++dst_ptr) {
                dt_int32 result = rounding_shift_right_away_from_zero(
                        round_mulh_saturate<dt_int32>(*x_ptr + bias, M),
                        offset);
                *dst_ptr = static_cast<dt_int8>(std::max<dt_int32>(
                        std::min<dt_int32>(result, maxv), minv));
            }
        }
    }
}

bool ElemwiseMultiTypeImpl::dispatch_fuse_add_rmulh_rshr(
        const ElemwiseOpParamN<6>& param, megdnn::dt_int8* dst) {
    BroadcastChannelInfo binfo;
    if (is_vector(param[0].layout) &&
        is_broadcasted_channel_like(param[1].layout, binfo) &&
        is_broadcasted_scalar(param[2].layout) &&
        is_broadcasted_scalar(param[3].layout) &&
        is_broadcasted_scalar(param[4].layout) &&
        is_broadcasted_scalar(param[5].layout)) {
        auto offset = param[3].ptr<dt_int8>()[0];
        auto minv = param[4].ptr<dt_int8>()[0];
        auto maxv = param[5].ptr<dt_int8>()[0];
        switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(stype, suffix)                                             \
    case DTypeTrait<stype>::enumv: {                                        \
        auto x_ptr = param[0].ptr<DTypeTrait<stype>::ctype>();              \
        auto b_ptr = param[1].ptr<DTypeTrait<stype>::ctype>();              \
        auto M = param[2].ptr<DTypeTrait<stype>::ctype>()[0];               \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                neon_fuse_add_rmulh_round_shr_saturate_bcast_1c11_##suffix( \
                        binfo.x, binfo.y, binfo.z, x_ptr, b_ptr, M, offset, \
                        minv, maxv, param.size, dst));                      \
        break;                                                              \
    }
            DISPATCH(dtype::Int16, int16)
            DISPATCH(dtype::Int32, int32)
            default:
                megdnn_throw("unreachable");
        }
        return true;
    }
    return false;
#undef DISPATCH
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
        const ElemwiseOpParamN<6>& param, megdnn::dt_int8* dst) {
    if (dispatch_fuse_add_rmulh_rshr(param, dst))
        return;
    fallback::ElemwiseMultiTypeImpl::
            on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(param, dst);
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
        const ElemwiseOpParamN<6>& param, megdnn::dt_int8* dst) {
    if (dispatch_fuse_add_rmulh_rshr(param, dst))
        return;
    fallback::ElemwiseMultiTypeImpl::
            on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(param, dst);
}

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<1>& param,
                                              const TensorND& dst,
                                              Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

#define DISPATCH_MODE(_src_dt, _dst_dt)                                      \
    switch (mode) {                                                          \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::RELU, ReluOp) \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::H_SWISH,      \
                             HSwishOp)                                       \
        default:                                                             \
            break;                                                           \
    }

#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt)                            \
    switch (mode) {                                                          \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::RELU, ReluOp) \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ABS, AbsOp)   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::SIGMOID,      \
                             SigmoidOp)                                      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::EXP, ExpOp)   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::TANH, TanhOp) \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FAST_TANH,    \
                             FastTanhOp)                                     \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::H_SWISH,      \
                             HSwishOp)                                       \
        default:                                                             \
            break;                                                           \
    }

#define DISPATCH()                                                            \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&            \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {                 \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8)       \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::Quantized8Asymm && \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {      \
        DISPATCH_QUANTIZED_MODE(dtype::Quantized8Asymm,                       \
                                dtype::Quantized8Asymm)                       \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&    \
               dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {          \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::QuantizedS8)                \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&    \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {      \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::Quantized8Asymm)            \
    }

    TensorND src = param[0];

    size_t nr_elems = src.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                \
    case _mode: {                                                         \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;            \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;            \
        thin_function<void(const src_ctype*, dst_ctype*, DType, DType,    \
                           size_t)>                                       \
                run = OpCallerUnary<_op<src_ctype, dst_ctype>, VEC>::run; \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                     \
                run(src.ptr<src_ctype>(), dst.ptr<dst_ctype>(),           \
                    src.layout.dtype, dst.layout.dtype, nr_elems));       \
        return;                                                           \
    }

    DISPATCH()

    fallback::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);

#undef DISPATCH_SINGLE_MODE
#undef DISPATCH
#undef DISPATCH_QUANTIZED_MODE
#undef DISPATCH_MODE
}

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<2>& param,
                                              const TensorND& dst,
                                              Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.enumv() ==
                          param[1].layout.dtype.enumv() &&
                  param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

#define DISPATCH_MODE(_src_dt, _dst_dt)                                       \
    switch (mode) {                                                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ADD, AddOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_RELU, \
                             FuseAddReluOp)                                   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_H_SWISH,                \
                             FuseAddHSwishOp)                                 \
        default:                                                              \
            break;                                                            \
    }

#if MEGDNN_AARCH64
#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt)                             \
    switch (mode) {                                                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ADD, AddOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MIN, MinOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MAX, MaxOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::SUB, SubOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MUL, MulOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::TRUE_DIV,      \
                             TrueDivOp)                                       \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_RELU, \
                             FuseAddReluOp)                                   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_SIGMOID,                \
                             FuseAddSigmoidOp)                                \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_TANH, \
                             FuseAddTanhOp)                                   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_H_SWISH,                \
                             FuseAddHSwishOp)                                 \
        default:                                                              \
            break;                                                            \
    }
#else
#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt)                             \
    switch (mode) {                                                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ADD, AddOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MIN, MinOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MAX, MaxOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::SUB, SubOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MUL, MulOp)    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_RELU, \
                             FuseAddReluOp)                                   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_SIGMOID,                \
                             FuseAddSigmoidOp)                                \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_TANH, \
                             FuseAddTanhOp)                                   \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_H_SWISH,                \
                             FuseAddHSwishOp)                                 \
        default:                                                              \
            break;                                                            \
    }
#endif

#define DISPATCH()                                                            \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&           \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {                 \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::QuantizedS8)                \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&    \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {      \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::Quantized8Asymm)            \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&     \
               dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {          \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8)       \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::Quantized8Asymm && \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {      \
        DISPATCH_QUANTIZED_MODE(dtype::Quantized8Asymm,                       \
                                dtype::Quantized8Asymm)                       \
    }

    TensorND src0 = param[0];
    TensorND src1 = param[1];

    //! VEC + VEC
    if (is_vector(src0.layout) && is_vector(src1.layout)) {
        size_t nr_elems = src0.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                     \
    case _mode: {                                                              \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                 \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                 \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,     \
                           DType, DType, DType, size_t)>                       \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>, VEC_VEC>::run; \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),              \
                    dst.ptr<dst_ctype>(), src0.layout.dtype,                   \
                    src1.layout.dtype, dst.layout.dtype, nr_elems));           \
        return;                                                                \
    }

        DISPATCH()

#undef DISPATCH_SINGLE_MODE
    }

    //! VEC + SCALAR
    {
        bool normal_case =
                is_vector(src0.layout) && is_broadcasted_scalar(src1.layout);
        bool swap_case = false;
        bool commutable = false;
        if (mode != Elemwise::Mode::SUB && mode != Elemwise::Mode::TRUE_DIV)
            commutable = true;
        if (!normal_case && commutable) {
            swap_case = is_vector(src1.layout) &&
                        is_broadcasted_scalar(src0.layout);
        }
        if (normal_case || swap_case) {
            auto &lhs = src0, &rhs = src1;
            if (swap_case)
                std::swap(lhs, rhs);
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                  \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype, dst_ctype*,   \
                           DType, DType, DType, size_t)>                    \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>,             \
                                     VEC_SCALAR>::run;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>()[0],            \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, src0.layout.total_nr_elems()));           \
        return;                                                             \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! SCALAR + VEC
        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_scalar(src0.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                  \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype, const src_ctype*, dst_ctype*,   \
                           DType, DType, DType, size_t)>                    \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>,             \
                                     SCALAR_VEC>::run;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>()[0], src1.ptr<src_ctype>(),            \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, src1.layout.total_nr_elems()));           \
        return;                                                             \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    //! VEC + BCAST101
    {
        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src0.layout) &&
                           is_broadcasted_channel_like(src1.layout, binfo);
        bool swap_case = false;
        bool commutable = false;
        if (mode != Elemwise::Mode::SUB && mode != Elemwise::Mode::TRUE_DIV)
            commutable = true;
        if (!normal_case && commutable) {
            swap_case = is_vector(src1.layout) &&
                        is_broadcasted_channel_like(src0.layout, binfo);
        }
        if (normal_case || swap_case) {
            auto &lhs = src0, &rhs = src1;
            if (swap_case)
                std::swap(lhs, rhs);
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                  \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,  \
                           DType, DType, DType, size_t, size_t, size_t)>    \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>,             \
                                     VEC_BCAST101>::run;                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, binfo.x, binfo.y, binfo.z));              \
        return;                                                             \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! BCAST101 + VEC : only for SUB or TRUE_DIV
        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_channel_like(src0.layout, binfo)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                  \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,  \
                           DType, DType, DType, size_t, size_t, size_t)>    \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>,             \
                                     BCAST101_VEC>::run;                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, binfo.x, binfo.y, binfo.z));              \
        return;                                                             \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    //! VEC + BCAST101x4
    {
        BroadcastChannelInfo binfo;
        if (is_vector(src0.layout) &&
            is_broadcastedx_channel_like<4>(src1.layout, binfo)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                  \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,  \
                           DType, DType, DType, size_t, size_t, size_t,     \
                           size_t)>                                         \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>,             \
                                     VEC_BCAST101x4>::run;                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z));  \
        return;                                                             \
    }

            size_t batch_size =
                    src0.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! BCAST101x + VEC
        if (is_vector(src1.layout) &&
            is_broadcastedx_channel_like<4>(src0.layout, binfo)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                  \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,  \
                           DType, DType, DType, size_t, size_t, size_t,     \
                           size_t)>                                         \
                run = OpCallerBinary<_op<src_ctype, dst_ctype>,             \
                                     BCAST101x4_VEC>::run;                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z));  \
        return;                                                             \
    }

            size_t batch_size =
                    src1.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    fallback::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);

#undef DISPATCH_MODE
#undef DISPATCH_QUANTIZED_MODE
#undef DISPATCH
}

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<3>& param,
                                              const TensorND& dst,
                                              Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.enumv() == param[1].layout.dtype.enumv() &&
            param[0].layout.dtype.enumv() == param[2].layout.dtype.enumv() &&
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt)                             \
    switch (mode) {                                                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_MUL_ADD3, \
                             FuseMulAdd3Op)                                   \
        default:                                                              \
            break;                                                            \
    }

#define DISPATCH()                                                            \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&            \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {                 \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8)       \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::Quantized8Asymm && \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {      \
        DISPATCH_QUANTIZED_MODE(dtype::Quantized8Asymm,                       \
                                dtype::Quantized8Asymm)                       \
    }

    TensorND src0 = param[0];
    TensorND src1 = param[1];
    TensorND src2 = param[2];

    //! VEC + VEC + VEC
    if (is_vector(src0.layout) && is_vector(src1.layout) &&
        is_vector(src2.layout)) {
        size_t nr_elems = src0.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                    \
    case _mode: {                                                             \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                \
        thin_function<void(const src_ctype*, const src_ctype*,                \
                           const src_ctype*, dst_ctype*, DType, DType, DType, \
                           DType, size_t)>                                    \
                run = OpCallerTernary<_op<src_ctype, dst_ctype>,              \
                                      VEC_VEC_VEC>::run;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                         \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),             \
                    src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),              \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,  \
                    dst.layout.dtype, nr_elems));                             \
        return;                                                               \
    }

        DISPATCH()

#undef DISPATCH_SINGLE_MODE
    }

    //! VEC + VEC + SCALAR
    if (is_vector(src0.layout) && is_vector(src1.layout) &&
        is_broadcasted_scalar(src2.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                   \
    case _mode: {                                                            \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;               \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;               \
        thin_function<void(const src_ctype*, const src_ctype*,               \
                           const src_ctype, dst_ctype*, DType, DType, DType, \
                           DType, size_t)>                                   \
                run = OpCallerTernary<_op<src_ctype, dst_ctype>,             \
                                      VEC_VEC_SCALAR>::run;                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),            \
                    src2.ptr<src_ctype>()[0], dst.ptr<dst_ctype>(),          \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype, \
                    dst.layout.dtype, src0.layout.total_nr_elems()));        \
        return;                                                              \
    }

        DISPATCH()

#undef DISPATCH_SINGLE_MODE
    }

    //! BCAST101 + VEC + BCAST101
    {
        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src1.layout) &&
                           is_broadcasted_channel_like(src0.layout, binfo) &&
                           src0.layout.eq_shape(src2.layout);
        if (normal_case) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                    \
    case _mode: {                                                             \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                \
        thin_function<void(const src_ctype*, const src_ctype*,                \
                           const src_ctype*, dst_ctype*, DType, DType, DType, \
                           DType, size_t, size_t, size_t)>                    \
                run = OpCallerTernary<_op<src_ctype, dst_ctype>,              \
                                      BCAST101_VEC_BCAST101>::run;            \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                         \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),             \
                    src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),              \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,  \
                    dst.layout.dtype, binfo.x, binfo.y, binfo.z));            \
        return;                                                               \
    }

            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    //! VEC + BCAST101x4 + VEC
    {
        BroadcastChannelInfo binfo;
        if (is_vector(src0.layout) &&
            is_broadcastedx_channel_like<4>(src1.layout, binfo) &&
            src0.layout.eq_shape(src2.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                     \
    case _mode: {                                                              \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                 \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                 \
        thin_function<void(const src_ctype*, const src_ctype*,                 \
                           const src_ctype*, dst_ctype*, DType, DType, DType,  \
                           DType, size_t, size_t, size_t, size_t)>             \
                run = OpCallerTernary<_op<src_ctype, dst_ctype>,               \
                                      VEC_BCAST101x4_VEC>::run;                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),              \
                    src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),               \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,   \
                    dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z)); \
        return;                                                                \
    }

            size_t batch_size =
                    src0.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }

        //! BCAST101x + VEC +BCAST101x
        if (is_vector(src1.layout) &&
            is_broadcastedx_channel_like<4>(src0.layout, binfo) &&
            src0.layout.eq_shape(src2.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op)                     \
    case _mode: {                                                              \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                 \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                 \
        thin_function<void(const src_ctype*, const src_ctype*,                 \
                           const src_ctype*, dst_ctype*, DType, DType, DType,  \
                           DType, size_t, size_t, size_t, size_t)>             \
                run = OpCallerTernary<_op<src_ctype, dst_ctype>,               \
                                      BCAST101x4_VEC_BCAST101x4>::run;         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),              \
                    src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),               \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,   \
                    dst.layout.dtype, batch_size, binfo.x, binfo.y, binfo.z)); \
        return;                                                                \
    }

            size_t batch_size =
                    src1.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
            DISPATCH()

#undef DISPATCH_SINGLE_MODE
        }
    }

    fallback::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);
#undef DISPATCH
#undef DISPATCH_QUANTIZED_MODE
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
