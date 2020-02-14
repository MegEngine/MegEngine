/**
 * \file dnn/src/x86/elemwise_multi_type/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/common/elemwise_multi_type/kern_defs.cuh"
#include "src/naive/handle.h"

#include "src/x86/elemwise_op.h"
#include "src/x86/simd_macro/immintrin.h"
#include "src/x86/utils.h"

using namespace megdnn;
using namespace x86;

#define DISPATCH_SIMD()                              \
    do {                                             \
        if (is_supported(SIMDType::AVX2)) {          \
            DISPATCH_DATA_TYPE(SIMDType::AVX2)       \
        } else if (is_supported(SIMDType::SSE4_2)) { \
            DISPATCH_DATA_TYPE(SIMDType::SSE4_2)     \
        }                                            \
    } while (0)

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<1>& param,
                                              const TensorND& dst,
                                              Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);
#if __x86_64__
#define DISPATCH_MODE(_src_dt, _dst_dt, _simd_type)                          \
    switch (mode) {                                                          \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::RELU, ReluOp, \
                             _simd_type)                                     \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::H_SWISH,      \
                             HSwishOp, _simd_type)                           \
        default:                                                             \
            break;                                                           \
    }

#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt, _simd_type)                \
    switch (mode) {                                                          \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::RELU, ReluOp, \
                             _simd_type)                                     \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ABS, AbsOp,   \
                             _simd_type)                                     \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::SIGMOID,      \
                             SigmoidOp, _simd_type)                          \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FAST_TANH,    \
                             FastTanhOp, _simd_type)                         \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::H_SWISH,      \
                             HSwishOp, _simd_type)                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::EXP, ExpOp,   \
                             _simd_type)                                     \
        default:                                                             \
            break;                                                           \
    }

#define DISPATCH_DATA_TYPE(_simd_type)                                         \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&             \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {                  \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8,        \
                                _simd_type)                                    \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::Quantized8Asymm &&  \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {       \
        DISPATCH_QUANTIZED_MODE(dtype::Quantized8Asymm,                        \
                                dtype::Quantized8Asymm, _simd_type)            \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&     \
               dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {           \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::QuantizedS8, _simd_type)     \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&     \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {       \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::Quantized8Asymm, _simd_type) \
    }
    
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)     \
    case _mode: {                                                          \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;             \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;             \
        thin_function<void(const src_ctype*, dst_ctype*, DType, DType,     \
                           size_t)>                                        \
                run = OpCallerUnary<_op<_simd_type, src_ctype, dst_ctype>, \
                                    _simd_type>::run;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                      \
                run(src.ptr<src_ctype>(), dst.ptr<dst_ctype>(),            \
                    src.layout.dtype, dst.layout.dtype, nr_elems));        \
        return;                                                            \
    }
    TensorND src = param[0];
    size_t nr_elems = src.layout.total_nr_elems();
    DISPATCH_SIMD();
#endif
    fallback::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);

#undef DISPATCH_SINGLE_MODE
#undef DISPATCH_DATA_TYPE
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
#if __x86_64__
#define DISPATCH_MODE(_src_dt, _dst_dt, _simd_type)                           \
    switch (mode) {                                                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ADD, AddOp,    \
                             _simd_type)                                      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_RELU, \
                             FuseAddReluOp, _simd_type)                       \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_H_SWISH,                \
                             FuseAddHSwishOp, _simd_type)                     \
        default:                                                              \
            break;                                                            \
    }

#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt, _simd_type)                 \
    switch (mode) {                                                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::ADD, AddOp,    \
                             _simd_type)                                      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MIN, MinOp,    \
                             _simd_type)                                      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MAX, MaxOp,    \
                             _simd_type)                                      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::SUB, SubOp,    \
                             _simd_type)                                      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::MUL, MulOp,    \
                             _simd_type)                                      \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_ADD_RELU, \
                             FuseAddReluOp, _simd_type)                       \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_SIGMOID,                \
                             FuseAddSigmoidOp, _simd_type)                    \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt,                                \
                             Elemwise::Mode::FUSE_ADD_H_SWISH,                \
                             FuseAddHSwishOp, _simd_type)                     \
        default:                                                              \
            break;                                                            \
    }

#define DISPATCH_DATA_TYPE(_simd_type)                                         \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&            \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {                  \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::QuantizedS8, _simd_type)     \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&     \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {       \
        DISPATCH_MODE(dtype::QuantizedS32, dtype::Quantized8Asymm, _simd_type) \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&      \
               dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {           \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8,        \
                                _simd_type)                                    \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::Quantized8Asymm &&  \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {       \
        DISPATCH_QUANTIZED_MODE(dtype::Quantized8Asymm,                        \
                                dtype::Quantized8Asymm, _simd_type)            \
    }

    TensorND src0 = param[0];
    TensorND src1 = param[1];

    //! VEC + VEC
    if (is_vector(src0.layout) && is_vector(src1.layout)) {
        size_t nr_elems = src0.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)         \
    case _mode: {                                                              \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                 \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                 \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,     \
                           DType, DType, DType, size_t)>                       \
                run = OpCallerBinary<_op<_simd_type, src_ctype, dst_ctype>, \
                                     _simd_type, VEC_VEC>::run;                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),              \
                    dst.ptr<dst_ctype>(), src0.layout.dtype,                   \
                    src1.layout.dtype, dst.layout.dtype, nr_elems));           \
        return;                                                                \
    }
        DISPATCH_SIMD();

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
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)      \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype, dst_ctype*,   \
                           DType, DType, DType, size_t)>                    \
                run = OpCallerBinary<_op<_simd_type, src_ctype, dst_ctype>, \
                                     _simd_type, VEC_SCALAR>::run;          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>()[0],            \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, src0.layout.total_nr_elems()));           \
        return;                                                             \
    }

            DISPATCH_SIMD();

#undef DISPATCH_SINGLE_MODE
        }

        //! SCALAR + VEC
        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_scalar(src0.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)      \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype, const src_ctype*, dst_ctype*,   \
                           DType, DType, DType, size_t)>                    \
                run = OpCallerBinary<_op<_simd_type, src_ctype, dst_ctype>, \
                                     _simd_type, SCALAR_VEC>::run;          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>()[0], src1.ptr<src_ctype>(),            \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, src1.layout.total_nr_elems()));           \
        return;                                                             \
    }
            DISPATCH_SIMD();

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
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)      \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,  \
                           DType, DType, DType, size_t, size_t, size_t)>    \
                run = OpCallerBinary<_op<_simd_type, src_ctype, dst_ctype>, \
                                     _simd_type, VEC_BCAST101>::run;        \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, binfo.x, binfo.y, binfo.z));              \
        return;                                                             \
    }

            DISPATCH_SIMD();

#undef DISPATCH_SINGLE_MODE
        }

        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_channel_like(src0.layout, binfo)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)      \
    case _mode: {                                                           \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;              \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;              \
        thin_function<void(const src_ctype*, const src_ctype*, dst_ctype*,  \
                           DType, DType, DType, size_t, size_t, size_t)>    \
                run = OpCallerBinary<_op<_simd_type, src_ctype, dst_ctype>, \
                                     _simd_type, BCAST101_VEC>::run;        \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(                                   \
                src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),               \
                dst.ptr<dst_ctype>(), src0.layout.dtype, src1.layout.dtype, \
                dst.layout.dtype, binfo.x, binfo.y, binfo.z));              \
        return;                                                             \
    }

            DISPATCH_SIMD();

#undef DISPATCH_SINGLE_MODE
        }
    }
#endif
    fallback::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);

#undef DISPATCH_MODE
#undef DISPATCH_QUANTIZED_MODE
#undef DISPATCH_DATA_TYPE
}

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<3>& param,
                                              const TensorND& dst,
                                              Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.enumv() == param[1].layout.dtype.enumv() &&
            param[0].layout.dtype.enumv() == param[2].layout.dtype.enumv() &&
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);
#if __x86_64__
#define DISPATCH_QUANTIZED_MODE(_src_dt, _dst_dt, _simd_type)                 \
    switch (mode) {                                                           \
        DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, Elemwise::Mode::FUSE_MUL_ADD3, \
                             FuseMulAdd3Op, _simd_type)                       \
        default:                                                              \
            break;                                                            \
    }

#define DISPATCH_DATA_TYPE(_simd_type)                                        \
    if (param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&            \
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {                 \
        DISPATCH_QUANTIZED_MODE(dtype::QuantizedS8, dtype::QuantizedS8,       \
                                _simd_type)                                   \
    } else if (param[0].layout.dtype.enumv() == DTypeEnum::Quantized8Asymm && \
               dst.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {      \
        DISPATCH_QUANTIZED_MODE(dtype::Quantized8Asymm,                       \
                                dtype::Quantized8Asymm, _simd_type)           \
    }

    TensorND src0 = param[0];
    TensorND src1 = param[1];
    TensorND src2 = param[2];

    //! VEC + VEC + VEC
    if (is_vector(src0.layout) && is_vector(src1.layout) &&
        is_vector(src2.layout)) {
        size_t nr_elems = src0.layout.total_nr_elems();
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)        \
    case _mode: {                                                             \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                \
        thin_function<void(const src_ctype*, const src_ctype*,                \
                           const src_ctype*, dst_ctype*, DType, DType, DType, \
                           DType, size_t)>                                    \
                run = OpCallerTernary<_op<_simd_type, src_ctype, dst_ctype>,  \
                                      _simd_type, VEC_VEC_VEC>::run;          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                         \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),             \
                    src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),              \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,  \
                    dst.layout.dtype, nr_elems));                             \
        return;                                                               \
    }
        DISPATCH_SIMD();
#undef DISPATCH_SINGLE_MODE
    }

    //! VEC + VEC + SCALAR
    if (is_vector(src0.layout) && is_vector(src1.layout) &&
        is_broadcasted_scalar(src2.layout)) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)       \
    case _mode: {                                                            \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;               \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;               \
        thin_function<void(const src_ctype*, const src_ctype*,               \
                           const src_ctype, dst_ctype*, DType, DType, DType, \
                           DType, size_t)>                                   \
                run = OpCallerTernary<_op<_simd_type, src_ctype, dst_ctype>, \
                                      _simd_type, VEC_VEC_SCALAR>::run;      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),            \
                    src2.ptr<src_ctype>()[0], dst.ptr<dst_ctype>(),          \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype, \
                    dst.layout.dtype, src0.layout.total_nr_elems()));        \
        return;                                                              \
    }
        DISPATCH_SIMD();
#undef DISPATCH_SINGLE_MODE
    }

    //! BCAST101 + VEC + BCAST101
    {
        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src1.layout) &&
                           is_broadcasted_channel_like(src0.layout, binfo) &&
                           src0.layout.eq_shape(src2.layout);
        if (normal_case) {
#define DISPATCH_SINGLE_MODE(_src_dt, _dst_dt, _mode, _op, _simd_type)         \
    case _mode: {                                                              \
        using src_ctype = typename DTypeTrait<_src_dt>::ctype;                 \
        using dst_ctype = typename DTypeTrait<_dst_dt>::ctype;                 \
        thin_function<void(const src_ctype*, const src_ctype*,                 \
                           const src_ctype*, dst_ctype*, DType, DType, DType,  \
                           DType, size_t, size_t, size_t)>                     \
                run = OpCallerTernary<_op<_simd_type, src_ctype, dst_ctype>,   \
                                      _simd_type, BCAST101_VEC_BCAST101>::run; \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                run(src0.ptr<src_ctype>(), src1.ptr<src_ctype>(),              \
                    src2.ptr<src_ctype>(), dst.ptr<dst_ctype>(),               \
                    src0.layout.dtype, src1.layout.dtype, src2.layout.dtype,   \
                    dst.layout.dtype, binfo.x, binfo.y, binfo.z));             \
        return;                                                                \
    }
        DISPATCH_SIMD();
#undef DISPATCH_SINGLE_MODE
        }
    }
#endif

    fallback::ElemwiseMultiTypeImpl::on_quantized_mode(param, dst, mode);
#undef DISPATCH_MODE
#undef DISPATCH_QUANTIZED_MODE
#undef DISPATCH_DATA_TYPE
}
#undef DISPATCH_SIMD
// vim: syntax=cpp.doxygen
