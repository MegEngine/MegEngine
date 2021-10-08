/**
 * \file dnn/src/arm_common/elemwise/binary/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/elemwise/binary/algo.h"
#include "src/arm_common/elemwise_op.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_elemwise_binary)

using namespace megdnn;
using namespace arm_common;

namespace {
static inline bool is_available_common(Elemwise::Mode mode) {
    /**
     * Fused sigmoid & tanh may be slower than the naive algo, because the
     * time used by neon function `exp_ps_f32` is decided by the input.
     */
    if (mode == Elemwise::Mode::FUSE_ADD_SIGMOID ||
        mode == Elemwise::Mode::FUSE_ADD_TANH) {
        return false;
    }

    return true;
}
}  // anonymous namespace

#if MEGDNN_AARCH64
#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id)             \
    auto mode = kern_param.mode;                                       \
    if (mode == Mode::MIN || mode == Mode::MAX || mode == Mode::ADD || \
        mode == Mode::SUB || mode == Mode::MUL || mode == Mode::POW || \
        mode == Mode::TRUE_DIV || mode == Mode::FUSE_ADD_RELU ||       \
        mode == Mode::FUSE_ADD_H_SWISH)                                \
        return true;
#else
#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id)             \
    auto mode = kern_param.mode;                                       \
    if (mode == Mode::MIN || mode == Mode::MAX || mode == Mode::ADD || \
        mode == Mode::SUB || mode == Mode::MUL || mode == Mode::POW || \
        mode == Mode::FUSE_ADD_RELU || mode == Mode::FUSE_ADD_H_SWISH) \
        return true;
#endif

#define DISPATCH_MODE_INT(_case, _type, _type_midout_id)                 \
    auto mode = kern_param.mode;                                         \
    if (mode == Mode::MIN || mode == Mode::MAX || mode == Mode::ADD ||   \
        mode == Mode::SUB || mode == Mode::MUL || mode == Mode::RMULH || \
        mode == Mode::FUSE_ADD_RELU)                                     \
        return true;

bool ElemwiseImpl::AlgoBinaryVecVec::is_available(const KernParam& kern_param) const {
    if (!is_available_common(kern_param.mode) ||
        (BcastType::VEC_VEC != kern_param.broad_cast_type))
        return false;

    auto& elparam = kern_param.binary_elparam;
    auto& src0 = elparam[0];

    //! exactly match [x, y] + [x, y]
    DISPATCH_TYPE("AlgoBinaryVecVec::is_available"_hash);

    return false;
}

bool ElemwiseImpl::AlgoBinaryVecScalar::is_available(
        const KernParam& kern_param) const {
    if (!is_available_common(kern_param.mode) ||
        ((BcastType::VEC_SCALAR != kern_param.broad_cast_type) &&
         (BcastType::SCALAR_VEC != kern_param.broad_cast_type)))
        return false;

    auto& elparam = kern_param.binary_elparam;
    auto& src0 = elparam[0];

    DISPATCH_TYPE("AlgoBinaryVecScalar::is_available"_hash);
    return false;
}

bool ElemwiseImpl::AlgoBinaryVecBcast101::is_available(
        const KernParam& kern_param) const {
    if (!is_available_common(kern_param.mode) ||
        ((BcastType::VEC_BCAST101 != kern_param.broad_cast_type) &&
         (BcastType::BCAST101_VEC != kern_param.broad_cast_type)))
        return false;

    auto& elparam = kern_param.binary_elparam;
    auto& src0 = elparam[0];

    DISPATCH_TYPE("AlgoBinaryVecBcast101::is_available"_hash);

    return false;
}

bool ElemwiseImpl::AlgoBinaryVecBcast111C::is_available(
        const KernParam& kern_param) const {
    if (!is_available_common(kern_param.mode) ||
        ((BcastType::VEC_BCAST111C != kern_param.broad_cast_type) &&
         (BcastType::BCAST111C_VEC != kern_param.broad_cast_type)))
        return false;

    auto& elparam = kern_param.binary_elparam;
    auto& src0 = elparam[0];

    DISPATCH_TYPE("AlgoBinaryVecBcast111C::is_available"_hash);

    return false;
}

bool ElemwiseImpl::AlgoBinaryVecBcast101xX::is_available(
        const KernParam& kern_param) const {
    if (!is_available_common(kern_param.mode) ||
        ((BcastType::VEC_BCAST101xX != kern_param.broad_cast_type) &&
         (BcastType::BCAST101xX_VEC != kern_param.broad_cast_type)))
        return false;

    auto& elparam = kern_param.binary_elparam;
    auto& src0 = elparam[0];

    DISPATCH_TYPE("AlgoBinaryVecBcast101xX::is_available"_hash);

    return false;
}

#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT

#if MEGDNN_AARCH64
#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id)                            \
    switch (kern_param.mode) {                                                        \
        DISPATCH_BINARY(MIN, _case, _type, _type_midout_id, MinOp);                   \
        DISPATCH_BINARY(MAX, _case, _type, _type_midout_id, MaxOp);                   \
        DISPATCH_BINARY(ADD, _case, _type, _type_midout_id, AddOp);                   \
        DISPATCH_BINARY(SUB, _case, _type, _type_midout_id, SubOp);                   \
        DISPATCH_BINARY(MUL, _case, _type, _type_midout_id, MulOp);                   \
        DISPATCH_BINARY(POW, _case, _type, _type_midout_id, PowOp);                   \
        DISPATCH_BINARY(TRUE_DIV, _case, _type, _type_midout_id, TrueDivOp);          \
        DISPATCH_BINARY(FUSE_ADD_RELU, _case, _type, _type_midout_id, FuseAddReluOp); \
        DISPATCH_BINARY(                                                              \
                FUSE_ADD_H_SWISH, _case, _type, _type_midout_id, FuseAddHSwishOp);    \
        default:                                                                      \
            megdnn_throw(ssprintf(                                                    \
                    "No avaiable algo find for: %d",                                  \
                    static_cast<int>(kern_param.mode)));                              \
    }
#else
#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id)                            \
    switch (kern_param.mode) {                                                        \
        DISPATCH_BINARY(MIN, _case, _type, _type_midout_id, MinOp);                   \
        DISPATCH_BINARY(MAX, _case, _type, _type_midout_id, MaxOp);                   \
        DISPATCH_BINARY(ADD, _case, _type, _type_midout_id, AddOp);                   \
        DISPATCH_BINARY(SUB, _case, _type, _type_midout_id, SubOp);                   \
        DISPATCH_BINARY(MUL, _case, _type, _type_midout_id, MulOp);                   \
        DISPATCH_BINARY(POW, _case, _type, _type_midout_id, PowOp);                   \
        DISPATCH_BINARY(FUSE_ADD_RELU, _case, _type, _type_midout_id, FuseAddReluOp); \
        DISPATCH_BINARY(                                                              \
                FUSE_ADD_H_SWISH, _case, _type, _type_midout_id, FuseAddHSwishOp);    \
        default:                                                                      \
            megdnn_throw(ssprintf(                                                    \
                    "No avaiable algo find for: %d",                                  \
                    static_cast<int>(kern_param.mode)));                              \
    }

#endif

#define DISPATCH_MODE_INT(_case, _type, _type_midout_id)                              \
    switch (kern_param.mode) {                                                        \
        DISPATCH_BINARY(MIN, _case, _type, _type_midout_id, MinOp);                   \
        DISPATCH_BINARY(MAX, _case, _type, _type_midout_id, MaxOp);                   \
        DISPATCH_BINARY(ADD, _case, _type, _type_midout_id, AddOp);                   \
        DISPATCH_BINARY(SUB, _case, _type, _type_midout_id, SubOp);                   \
        DISPATCH_BINARY(MUL, _case, _type, _type_midout_id, MulOp);                   \
        DISPATCH_BINARY(RMULH, _case, _type, _type_midout_id, RmulhOp);               \
        DISPATCH_BINARY(FUSE_ADD_RELU, _case, _type, _type_midout_id, FuseAddReluOp); \
        default:                                                                      \
            megdnn_throw(ssprintf(                                                    \
                    "No avaiable algo find for: %d",                                  \
                    static_cast<int>(kern_param.mode)));                              \
    }

void ElemwiseImpl::AlgoBinaryVecVec::exec(const KernParam& kern_param) const {
    auto& elparam = kern_param.binary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1];

    //! exactly match [x, y] + [x, y]
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                    \
    case Mode::_mode:                                                                 \
        MIDOUT_BEGIN(                                                                 \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                  \
                midout_iv(Mode::_mode), _type_midout_id) {                            \
            thin_function<void(                                                       \
                    const _type*, const _type*, _type*, DType, DType, DType, size_t)> \
                    run = OpCallerBinary<_op<_type, _type>, BcastType::VEC_VEC>::run; \
            MEGDNN_DISPATCH_CPU_KERN(                                                 \
                    static_cast<naive::HandleImpl*>(kern_param.handle),               \
                    run(static_cast<const _type*>(src0.raw_ptr),                      \
                        static_cast<const _type*>(src1.raw_ptr),                      \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,          \
                        src1.layout.dtype, dst.layout.dtype,                          \
                        src0.layout.total_nr_elems()));                               \
        }                                                                             \
        MIDOUT_END();                                                                 \
        return

    auto&& dst = *(kern_param.m_dst);
    DISPATCH_TYPE("AlgoBinaryVecVec::exec"_hash);

#undef DISPATCH_BINARY

    return;
}

void ElemwiseImpl::AlgoBinaryVecScalar::exec(const KernParam& kern_param) const {
    auto& elparam = kern_param.binary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1];
    auto&& dst = *(kern_param.m_dst);

    // Case 2: vector + scalar
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type*, const _type, _type*, DType, DType, DType, size_t)> \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::VEC_SCALAR>::run;          \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr),                     \
                        static_cast<const _type*>(src1.raw_ptr)[0],                  \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype,                         \
                        src0.layout.total_nr_elems()));                              \
        }                                                                            \
        MIDOUT_END();                                                                \
        return

    if (BcastType::VEC_SCALAR == kern_param.broad_cast_type) {
        DISPATCH_TYPE("AlgoBinaryVecScalar::exec_vec_sca"_hash);
    }
#undef DISPATCH_BINARY

    // scalar + vector
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type, const _type*, _type*, DType, DType, DType, size_t)> \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::SCALAR_VEC>::run;          \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr)[0],                  \
                        static_cast<const _type*>(src1.raw_ptr),                     \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype,                         \
                        src1.layout.total_nr_elems()));                              \
        }                                                                            \
        MIDOUT_END();                                                                \
        return

    if (BcastType::SCALAR_VEC == kern_param.broad_cast_type) {
        DISPATCH_TYPE("AlgoBinaryVecScalar::exec_sca_vec"_hash);
    }
#undef DISPATCH_BINARY

    return;
}

void ElemwiseImpl::AlgoBinaryVecBcast101::exec(const KernParam& kern_param) const {
    auto& elparam = kern_param.binary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1];
    auto&& dst = *(kern_param.m_dst);
    BroadcastChannelInfo binfo;

    // Case 3: BcastType::VEC + BCAST_101
    if (BcastType::VEC_BCAST101 == kern_param.broad_cast_type &&
        is_broadcasted_channel_like(src1.layout, binfo)) {
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type*, const _type*, _type*, DType, DType, DType, size_t, \
                    size_t, size_t)>                                                 \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::VEC_BCAST101>::run;        \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr),                     \
                        static_cast<const _type*>(src1.raw_ptr),                     \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype, binfo.x, binfo.y,       \
                        binfo.z));                                                   \
        }                                                                            \
        MIDOUT_END();                                                                \
        return

        DISPATCH_TYPE("AlgoBinaryVecBcast101::exec_vec_b"_hash);

#undef DISPATCH_BINARY
    }

    // BCAST_101 + BcastType::VEC
    if (BcastType::BCAST101_VEC == kern_param.broad_cast_type &&
        is_broadcasted_channel_like(src0.layout, binfo)) {
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type*, const _type*, _type*, DType, DType, DType, size_t, \
                    size_t, size_t)>                                                 \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::BCAST101_VEC>::run;        \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr),                     \
                        static_cast<const _type*>(src1.raw_ptr),                     \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype, binfo.x, binfo.y,       \
                        binfo.z));                                                   \
        }                                                                            \
        MIDOUT_END();                                                                \
        return

        DISPATCH_TYPE("AlgoBinaryVecBcast101::exec_b_vec"_hash);

#undef DISPATCH_BINARY
    }
    return;
}

void ElemwiseImpl::AlgoBinaryVecBcast111C::exec(const KernParam& kern_param) const {
    auto& elparam = kern_param.binary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1];
    auto&& dst = *(kern_param.m_dst);
    BroadcastChannelInfo binfo;

    // Case extra: BcastType::VEC + BCAST_111C
    if (BcastType::VEC_BCAST111C == kern_param.broad_cast_type &&
        is_NHWC_broadcasted_channel_like(src1.layout, binfo)) {
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type*, const _type*, _type*, DType, DType, DType, size_t, \
                    size_t, size_t)>                                                 \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::VEC_BCAST111C>::run;       \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr),                     \
                        static_cast<const _type*>(src1.raw_ptr),                     \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype, binfo.x, binfo.y,       \
                        binfo.z));                                                   \
        }                                                                            \
        MIDOUT_END();                                                                \
        return

        DISPATCH_TYPE("AlgoBinaryVecBcast111C::exec_vec_b"_hash);

#undef DISPATCH_BINARY
    }

    // BCAST_111C + BcastType::VEC
    if (BcastType::BCAST111C_VEC == kern_param.broad_cast_type &&
        is_NHWC_broadcasted_channel_like(src0.layout, binfo)) {
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type*, const _type*, _type*, DType, DType, DType, size_t, \
                    size_t, size_t)>                                                 \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::BCAST111C_VEC>::run;       \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr),                     \
                        static_cast<const _type*>(src1.raw_ptr),                     \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype, binfo.x, binfo.y,       \
                        binfo.z));                                                   \
        }                                                                            \
        MIDOUT_END();                                                                \
        return

        DISPATCH_TYPE("AlgoBinaryVecBcast111C::exec_b_vec"_hash);

#undef DISPATCH_BINARY
    }
    return;
}

void ElemwiseImpl::AlgoBinaryVecBcast101xX::exec(const KernParam& kern_param) const {
    auto& elparam = kern_param.binary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1];
    auto&& dst = *(kern_param.m_dst);
    BroadcastChannelInfo binfo;

    //  BcastType::VEC + BCAST_101X
    if (BcastType::VEC_BCAST101xX == kern_param.broad_cast_type) {
        megdnn_assert(
                is_broadcastedx_channel_like<4>(src1.layout, binfo) ||
                        is_broadcastedx_channel_like<8>(src1.layout, binfo),
                "only nchw44 and nchw88 supported");
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type*, const _type*, _type*, DType, DType, DType, size_t, \
                    size_t, size_t, size_t)>                                         \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::VEC_BCAST101xX>::run;      \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr),                     \
                        static_cast<const _type*>(src1.raw_ptr),                     \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype, batch_size, binfo.x,    \
                        binfo.y, binfo.z));                                          \
        }                                                                            \
        MIDOUT_END();                                                                \
        return
        size_t batch_size = src0.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
        DISPATCH_TYPE("AlgoBinaryVecBcast101xX::exec_vec_b"_hash);

#undef DISPATCH_BINARY
    }

    // BCAST_101x + BcastType::VEC
    if (BcastType::BCAST101xX_VEC == kern_param.broad_cast_type) {
        megdnn_assert(
                is_broadcastedx_channel_like<4>(src0.layout, binfo) ||
                        is_broadcastedx_channel_like<8>(src0.layout, binfo),
                "only nchw44 and nchw88 supported");
#define DISPATCH_BINARY(_mode, _case, _type, _type_midout_id, _op)                   \
    case Mode::_mode:                                                                \
        MIDOUT_BEGIN(                                                                \
                megdnn_arm_common_elemwise_binary, midout_iv(_case),                 \
                midout_iv(Mode::_mode), _type_midout_id) {                           \
            thin_function<void(                                                      \
                    const _type*, const _type*, _type*, DType, DType, DType, size_t, \
                    size_t, size_t, size_t)>                                         \
                    run = OpCallerBinary<                                            \
                            _op<_type, _type>, BcastType::BCAST101xX_VEC>::run;      \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    static_cast<naive::HandleImpl*>(kern_param.handle),              \
                    run(static_cast<const _type*>(src0.raw_ptr),                     \
                        static_cast<const _type*>(src1.raw_ptr),                     \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,         \
                        src1.layout.dtype, dst.layout.dtype, batch_size, binfo.x,    \
                        binfo.y, binfo.z));                                          \
        }                                                                            \
        MIDOUT_END();                                                                \
        return
        size_t batch_size = src1.layout.shape[0] / (binfo.x * binfo.y * binfo.z);

        DISPATCH_TYPE("AlgoBinaryVecBcast101xX::exec_b_vec"_hash);

#undef DISPATCH_BINARY
    }
    return;
}

#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT

// vim: syntax=cpp.doxygen
