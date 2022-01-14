/**
 * \file dnn/src/naive/elemwise_multi_type/opr_impl_3.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/elemwise_multi_type/kern_defs.cuh"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

template <typename KernImpl, typename src_ctype, typename dst_ctype>
void ElemwiseMultiTypeImpl::dispatch_add_qint_op(
        const ElemwiseOpParamN<1>& param, const TensorND& dst_tensor) {
    auto src = param[0];
    auto size = param.size;
    auto work = [src, size, dst_tensor]() {
        auto iA = tensor_iter_valonly<src_ctype>(src).begin();
        auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();

        auto param0 = src.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
        auto dst_param =
                dst_tensor.layout.dtype.param<typename DTypeTrait<dst_ctype>::dtype>();
        for (size_t i = 0; i < size; i++) {
            src_ctype a = *iA;
            *pD = dst_param.quantize(KernImpl::apply(param0.dequantize(a)));
            ++iA;
            ++pD;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

template <typename KernImpl, typename src_ctype, typename dst_ctype>
void ElemwiseMultiTypeImpl::dispatch_add_qint_op(
        const ElemwiseOpParamN<2>& param, const TensorND& dst_tensor) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto work = [src0, src1, size, dst_tensor]() {
        // This is needed as these iterators are captured as const value.
        auto iA = tensor_iter_valonly<src_ctype>(src0).begin();
        auto iB = tensor_iter_valonly<src_ctype>(src1).begin();
        auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();
        auto param0 = src0.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
        auto param1 = src1.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
        auto dst_param =
                dst_tensor.layout.dtype.param<typename DTypeTrait<dst_ctype>::dtype>();
        for (size_t i = 0; i < size; i++) {
            src_ctype a = *iA;
            src_ctype b = *iB;
            *pD = dst_param.quantize(
                    KernImpl::apply(param0.dequantize(a), param1.dequantize(b)));
            ++iA;
            ++iB;
            ++pD;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

template <typename KernImpl, typename src_ctype, typename dst_ctype>
void ElemwiseMultiTypeImpl::dispatch_add_qint_op(
        const ElemwiseOpParamN<3>& param, const TensorND& dst_tensor) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst_tensor]() {
        // This is needed as these iterators are captured as const value.
        auto iA = tensor_iter_valonly<src_ctype>(src0).begin();
        auto iB = tensor_iter_valonly<src_ctype>(src1).begin();
        auto iC = tensor_iter_valonly<src_ctype>(src2).begin();
        auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();
        auto param0 = src0.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
        auto param1 = src1.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
        auto param2 = src2.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
        auto dst_param =
                dst_tensor.layout.dtype.param<typename DTypeTrait<dst_ctype>::dtype>();
        for (size_t i = 0; i < size; i++) {
            src_ctype a = *iA;
            src_ctype b = *iB;
            src_ctype c = *iC;
            *pD = dst_param.quantize(KernImpl::apply(
                    param0.dequantize(a), param1.dequantize(b), param2.dequantize(c)));
            ++iA;
            ++iB;
            ++iC;
            ++pD;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

template <typename KernImpl, typename src_ctype, typename ElemParam>
void ElemwiseMultiTypeImpl::dispatch_add_qint_op_dst(
        const ElemParam& param, const TensorND& dst) {
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                     \
    case DTypeTrait<_dt>::enumv:                                                    \
        dispatch_add_qint_op<KernImpl, src_ctype, typename DTypeTrait<_dt>::ctype>( \
                param, dst);                                                        \
        break;
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
#undef cb

        default:
            megdnn_assert(
                    0, "not support %s %s\n", param[0].layout.dtype.name(),
                    dst.layout.dtype.name());
    }
}

template <typename KernImpl, typename ElemParam>
void ElemwiseMultiTypeImpl::dispatch_qint_op_dtype(
        const ElemParam& param, const TensorND& dst) {
    switch (param[0].layout.dtype.enumv()) {
#define cb(_dt)                                                                    \
    case DTypeTrait<_dt>::enumv:                                                   \
        dispatch_add_qint_op_dst<                                                  \
                KernImpl, typename DTypeTrait<_dt>::ctype, ElemParam>(param, dst); \
        break;
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
#undef cb

        default:
            megdnn_assert_internal(0);
    }
}

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<1>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

    switch (mode) {
#define DISPATCH(_mode)                                                        \
    case Elemwise::Mode::_mode: {                                              \
        typedef ElemwiseKern<                                                  \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode, float> \
                KernImpl;                                                      \
        dispatch_qint_op_dtype<KernImpl, ElemwiseOpParamN<1>>(param, dst);     \
        break;                                                                 \
    }

        DISPATCH(RELU);
        DISPATCH(ABS);
        DISPATCH(ACOS);
        DISPATCH(ASIN);
        DISPATCH(CEIL);
        DISPATCH(COS);
        DISPATCH(EXP);
        DISPATCH(EXPM1);
        DISPATCH(FLOOR);
        DISPATCH(LOG);
        DISPATCH(LOG1P);
        DISPATCH(NEGATE);
        DISPATCH(SIGMOID);
        DISPATCH(SIN);
        DISPATCH(TANH);
        DISPATCH(FAST_TANH);
        DISPATCH(ROUND);
        DISPATCH(ERF);
        DISPATCH(ERFINV);
        DISPATCH(ERFC);
        DISPATCH(ERFCINV);
        DISPATCH(H_SWISH);
#undef DISPATCH
        default:
            megdnn_assert_internal(0);
    }
}

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<2>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.enumv() == param[1].layout.dtype.enumv() &&
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

    switch (mode) {
#define DISPATCH(_mode)                                                        \
    case Elemwise::Mode::_mode: {                                              \
        typedef ElemwiseKern<                                                  \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode, float> \
                KernImpl;                                                      \
        dispatch_qint_op_dtype<KernImpl, ElemwiseOpParamN<2>>(param, dst);     \
        break;                                                                 \
    }

        DISPATCH(ABS_GRAD);
        DISPATCH(ADD);
        DISPATCH(FLOOR_DIV);
        DISPATCH(MAX);
        DISPATCH(MIN);
        DISPATCH(MOD);
        DISPATCH(MUL);
        DISPATCH(POW);
        DISPATCH(SIGMOID_GRAD);
        DISPATCH(SUB);
        DISPATCH(SWITCH_GT0);
        DISPATCH(TANH_GRAD);
        DISPATCH(TRUE_DIV);
        DISPATCH(LOG_SUM_EXP);

        DISPATCH(LT);
        DISPATCH(LEQ);
        DISPATCH(EQ);

        DISPATCH(FUSE_ADD_RELU);
        DISPATCH(FUSE_ADD_SIGMOID);
        DISPATCH(FUSE_ADD_TANH);
        DISPATCH(FAST_TANH_GRAD);
        DISPATCH(ATAN2);
        DISPATCH(H_SWISH_GRAD);
        DISPATCH(FUSE_ADD_H_SWISH);
#undef DISPATCH
        default:
            megdnn_assert_internal(0);
    }
}

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<3>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED &&
            param[0].layout.dtype.category() == param[1].layout.dtype.category() &&
            param[0].layout.dtype.category() == param[2].layout.dtype.category());
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

    switch (mode) {
#define DISPATCH(_mode)                                                        \
    case Elemwise::Mode::_mode: {                                              \
        typedef ElemwiseKern<                                                  \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode, float> \
                KernImpl;                                                      \
        dispatch_qint_op_dtype<KernImpl, ElemwiseOpParamN<3>>(param, dst);     \
        break;                                                                 \
    }

        DISPATCH(FUSE_MUL_ADD3);
        DISPATCH(COND_LEQ_MOV);
#undef DISPATCH
        default:
            megdnn_assert_internal(0);
    }
}

// vim: syntax=cpp.doxygen
