/**
 * \file dnn/src/arm_common/elemwise/ternary/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/elemwise/ternary/algo.h"
#include "src/arm_common/elemwise_op.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_elemwise_ternary)

using namespace megdnn;
using namespace arm_common;

#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id) \
    auto mode = kern_param.mode;                           \
    if (mode == Mode::FUSE_MUL_ADD3)                       \
        return true;
#define DISPATCH_MODE_INT DISPATCH_MODE_FLOAT

#define DECL_AVAILABLE(case, type)                                       \
    bool ElemwiseImpl::AlgoTernaryFma3##case ::is_available(             \
            const KernParam& kern_param) const {                         \
        if (type == kern_param.broad_cast_type) {                        \
            auto& elparam = kern_param.ternary_elparam;                  \
            auto& src0 = elparam[0];                                     \
            DISPATCH_TYPE("AlgoTernaryFma3::is_available" #case##_hash); \
        }                                                                \
        return false;                                                    \
    }

DECL_AVAILABLE(VecVecVec, BcastType::VEC_VEC_VEC);
DECL_AVAILABLE(VecVecScalar, BcastType::VEC_VEC_SCALAR);
DECL_AVAILABLE(Bcast101VecBcast101, BcastType::BCAST101_VEC_BCAST101);
DECL_AVAILABLE(VecBcast101Vec, BcastType::VEC_BCAST101_VEC);
DECL_AVAILABLE(VecScalarVec, BcastType::VEC_SCALAR_VEC);
DECL_AVAILABLE(VecScalarScalar, BcastType::VEC_SCALAR_SCALAR);
#undef DECL_CB
#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT

#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id)             \
    switch (kern_param.mode) {                                         \
        DISPATCH_TERNARY(FUSE_MUL_ADD3, _case, _type, _type_midout_id, \
                         FuseMulAdd3Op);                               \
        default:                                                       \
            megdnn_throw(ssprintf("No avaiable algo find for: %d",     \
                                  static_cast<int>(kern_param.mode))); \
    }
#define DISPATCH_MODE_INT DISPATCH_MODE_FLOAT
void ElemwiseImpl::AlgoTernaryFma3VecVecVec::exec(
        const KernParam& kern_param) const {
    auto& elparam = kern_param.ternary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1], &src2 = elparam[2];

    // Case 1: shape of (src0, src2) and src1 are exactly match
#define DISPATCH_TERNARY(_mode, _case, _type, _type_midout_id, _op)          \
    case Mode::_mode:                                                        \
        MIDOUT_BEGIN(megdnn_arm_common_elemwise_ternary, midout_iv(_case),   \
                     midout_iv(Mode::_mode), _type_midout_id) {              \
            thin_function<void(const _type*, const _type*, const _type*,     \
                               _type*, DType, DType, DType, DType, size_t)>  \
                    run = OpCallerTernary<_op<_type, _type>,                 \
                                          BcastType::VEC_VEC_VEC>::run;      \
            MEGDNN_DISPATCH_CPU_KERN(                                        \
                    static_cast<naive::HandleImpl*>(kern_param.handle),      \
                    run(static_cast<const _type*>(src0.raw_ptr),             \
                        static_cast<const _type*>(src1.raw_ptr),             \
                        static_cast<const _type*>(src2.raw_ptr),             \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype, \
                        src1.layout.dtype, src2.layout.dtype,                \
                        dst.layout.dtype, src0.layout.total_nr_elems()));    \
        }                                                                    \
        MIDOUT_END();                                                        \
        return

    auto&& dst = *(kern_param.m_dst);
    DISPATCH_TYPE("AlgoTernaryFma3VecVecVec::exec"_hash);
#undef DISPATCH_TERNARY

    return;
}
void ElemwiseImpl::AlgoTernaryFma3VecVecScalar::exec(
        const KernParam& kern_param) const {
    auto& elparam = kern_param.ternary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1], &src2 = elparam[2];

    // Case 2: (src2 is a scalar) && (src0 and src1 has the same shape)
#define DISPATCH_TERNARY(_mode, _case, _type, _type_midout_id, _op)          \
    case Mode::_mode:                                                        \
        MIDOUT_BEGIN(megdnn_arm_common_elemwise_ternary, midout_iv(_case),   \
                     midout_iv(Mode::_mode), _type_midout_id) {              \
            thin_function<void(const _type*, const _type*, const _type,      \
                               _type*, DType, DType, DType, DType, size_t)>  \
                    run = OpCallerTernary<_op<_type, _type>,                 \
                                          BcastType::VEC_VEC_SCALAR>::run;   \
            MEGDNN_DISPATCH_CPU_KERN(                                        \
                    static_cast<naive::HandleImpl*>(kern_param.handle),      \
                    run(static_cast<const _type*>(src0.raw_ptr),             \
                        static_cast<const _type*>(src1.raw_ptr),             \
                        static_cast<const _type*>(src2.raw_ptr)[0],          \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype, \
                        src1.layout.dtype, src2.layout.dtype,                \
                        dst.layout.dtype, src0.layout.total_nr_elems()));    \
        }                                                                    \
        MIDOUT_END();                                                        \
        return

    auto&& dst = *(kern_param.m_dst);
    DISPATCH_TYPE("AlgoTernaryFma3VecVecScalar::exec"_hash);
#undef DISPATCH_TERNARY

    return;
}
void ElemwiseImpl::AlgoTernaryFma3Bcast101VecBcast101::exec(
        const KernParam& kern_param) const {
    auto& elparam = kern_param.ternary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1], &src2 = elparam[2];

    // Case 3: shape of src0 and src2 is {1, C, 1, 1}
    BroadcastChannelInfo binfo;
    is_broadcasted_channel_like(src0.layout, binfo);
#define DISPATCH_TERNARY(_mode, _case, _type, _type_midout_id, _op)          \
    case Mode::_mode:                                                        \
        MIDOUT_BEGIN(megdnn_arm_common_elemwise_ternary, midout_iv(_case),   \
                     midout_iv(Mode::_mode), _type_midout_id) {              \
            thin_function<void(const _type*, const _type*, const _type*,     \
                               _type*, DType, DType, DType, DType, size_t,   \
                               size_t, size_t)>                              \
                    run = OpCallerTernary<                                   \
                            _op<_type, _type>,                               \
                            BcastType::BCAST101_VEC_BCAST101>::run;          \
            MEGDNN_DISPATCH_CPU_KERN(                                        \
                    static_cast<naive::HandleImpl*>(kern_param.handle),      \
                    run(static_cast<const _type*>(src0.raw_ptr),             \
                        static_cast<const _type*>(src1.raw_ptr),             \
                        static_cast<const _type*>(src2.raw_ptr),             \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype, \
                        src1.layout.dtype, src2.layout.dtype,                \
                        dst.layout.dtype, binfo.x, binfo.y, binfo.z));       \
        }                                                                    \
        MIDOUT_END();                                                        \
        return

    auto&& dst = *(kern_param.m_dst);
    DISPATCH_TYPE("AlgoTernaryFma3Bcast101VecBcast101::exec"_hash);
#undef DISPATCH_TERNARY

    return;
}
void ElemwiseImpl::AlgoTernaryFma3VecBcast101Vec::exec(
        const KernParam& kern_param) const {
    auto& elparam = kern_param.ternary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1], &src2 = elparam[2];

    // Case 4: shape of src1 is {1, C, 1, 1}, and src0 and src2 are contig
    BroadcastChannelInfo binfo;
    is_broadcasted_channel_like(src1.layout, binfo);
#define DISPATCH_TERNARY(_mode, _case, _type, _type_midout_id, _op)          \
    case Mode::_mode:                                                        \
        MIDOUT_BEGIN(megdnn_arm_common_elemwise_ternary, midout_iv(_case),   \
                     midout_iv(Mode::_mode), _type_midout_id) {              \
            thin_function<void(const _type*, const _type*, const _type*,     \
                               _type*, DType, DType, DType, DType, size_t,   \
                               size_t, size_t)>                              \
                    run = OpCallerTernary<_op<_type, _type>,                 \
                                          BcastType::VEC_BCAST101_VEC>::run; \
            MEGDNN_DISPATCH_CPU_KERN(                                        \
                    static_cast<naive::HandleImpl*>(kern_param.handle),      \
                    run(static_cast<const _type*>(src0.raw_ptr),             \
                        static_cast<const _type*>(src1.raw_ptr),             \
                        static_cast<const _type*>(src2.raw_ptr),             \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype, \
                        src1.layout.dtype, src2.layout.dtype,                \
                        dst.layout.dtype, binfo.x, binfo.y, binfo.z));       \
        }                                                                    \
        MIDOUT_END();                                                        \
        return

    auto&& dst = *(kern_param.m_dst);
    DISPATCH_TYPE("AlgoTernaryFma3VecBcast101Vec::exec"_hash);
#undef DISPATCH_TERNARY

    return;
}
void ElemwiseImpl::AlgoTernaryFma3VecScalarVec::exec(
        const KernParam& kern_param) const {
    auto& elparam = kern_param.ternary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1], &src2 = elparam[2];

    // Case 5: (src1 is a scalar) && (src0 and src2 has the same shape)
#define DISPATCH_TERNARY(_mode, _case, _type, _type_midout_id, _op)          \
    case Mode::_mode:                                                        \
        MIDOUT_BEGIN(megdnn_arm_common_elemwise_ternary, midout_iv(_case),   \
                     midout_iv(Mode::_mode), _type_midout_id) {              \
            thin_function<void(const _type*, const _type, const _type*,      \
                               _type*, DType, DType, DType, DType, size_t)>  \
                    run = OpCallerTernary<_op<_type, _type>,                 \
                                          BcastType::VEC_SCALAR_VEC>::run;   \
            MEGDNN_DISPATCH_CPU_KERN(                                        \
                    static_cast<naive::HandleImpl*>(kern_param.handle),      \
                    run(static_cast<const _type*>(src0.raw_ptr),             \
                        static_cast<const _type*>(src1.raw_ptr)[0],          \
                        static_cast<const _type*>(src2.raw_ptr),             \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype, \
                        src1.layout.dtype, src2.layout.dtype,                \
                        dst.layout.dtype, src0.layout.total_nr_elems()));    \
        }                                                                    \
        MIDOUT_END();                                                        \
        return

    auto&& dst = *(kern_param.m_dst);
    DISPATCH_TYPE("AlgoTernaryFma3VecScalarVec::exec"_hash);
#undef DISPATCH_TERNARY

    return;
}
void ElemwiseImpl::AlgoTernaryFma3VecScalarScalar::exec(
        const KernParam& kern_param) const {
    auto& elparam = kern_param.ternary_elparam;
    auto &src0 = elparam[0], &src1 = elparam[1], &src2 = elparam[2];

    // Case 6: (src1 and src2 is scalar) && (src0 is vector)
#define DISPATCH_TERNARY(_mode, _case, _type, _type_midout_id, _op)            \
    case Mode::_mode:                                                          \
        MIDOUT_BEGIN(megdnn_arm_common_elemwise_ternary, midout_iv(_case),     \
                     midout_iv(Mode::_mode), _type_midout_id) {                \
            thin_function<void(const _type*, const _type, const _type, _type*, \
                               DType, DType, DType, DType, size_t)>            \
                    run = OpCallerTernary<_op<_type, _type>,                   \
                                          BcastType::VEC_SCALAR_SCALAR>::run;  \
            MEGDNN_DISPATCH_CPU_KERN(                                          \
                    static_cast<naive::HandleImpl*>(kern_param.handle),        \
                    run(static_cast<const _type*>(src0.raw_ptr),               \
                        static_cast<const _type*>(src1.raw_ptr)[0],            \
                        static_cast<const _type*>(src2.raw_ptr)[0],            \
                        static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,   \
                        src1.layout.dtype, src2.layout.dtype,                  \
                        dst.layout.dtype, src0.layout.total_nr_elems()));      \
        }                                                                      \
        MIDOUT_END();                                                          \
        return

    auto&& dst = *(kern_param.m_dst);
    DISPATCH_TYPE("AlgoTernaryFma3VecScalarScalar::exec"_hash);
#undef DISPATCH_TERNARY

    return;
}
#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT

// vim: syntax=cpp.doxygen
