/**
 * \file dnn/src/naive/elemwise_multi_type/opr_impl.cpp
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

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_int16x32x32x32(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_int16>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_int32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_int32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_int32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1) + (*i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_int16xf32xf32xf32(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_int16>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_float32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_float32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1) + (*i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_uint8xf32xf32xf32(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_uint8>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_float32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_float32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1) + (*i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_mul_int16xf32xf32(
        const ElemwiseOpParamN<2>& param, const TensorND& dst) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto work = [src0, src1, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_int16>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto dst_ptr = dst.ptr<dt_float32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1);
            ++i0;
            ++i1;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_iXxf32xf32xi8(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
    switch (param[0].layout.dtype.enumv()) {
#define cb(t)                  \
    case DTypeTrait<t>::enumv: \
        return dispatch_fma3_iXxf32xf32xi8<DTypeTrait<t>::ctype>(param, dst);
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
        default:
            megdnn_throw("unsupported src dtype");
    }
}

template <typename ctype>
void ElemwiseMultiTypeImpl::dispatch_fma3_iXxf32xf32xi8(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        elemwise_multi_type::Fma3iXxf32xf32xiYOp<ctype, dt_int8> op;
        auto i0 = tensor_iter_valonly<ctype>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_float32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_int8>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = op(*i0, *i1, *i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi8(
        const ElemwiseOpParamN<2>& param, const TensorND& dst) {
    switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                                       \
    case DTypeTrait<t>::enumv:                                                      \
        return dispatch_round_shr_saturate_iXxi8xiX<DTypeTrait<t>::ctype, dt_int8>( \
                param, dst);
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
        default:
            megdnn_throw("unsupported src dtype");
    }
}

template <typename ctype, typename dst_ctype>
void ElemwiseMultiTypeImpl::dispatch_round_shr_saturate_iXxi8xiX(
        const ElemwiseOpParamN<2>& param, const TensorND& dst) {
    auto src0 = param[0];
    auto src1 = param[1];
    auto size = param.size;
    auto work = [src0, src1, size, dst]() {
        // This is needed as these iterators are captured as const value.
        auto iA = tensor_iter_valonly<ctype>(src0).begin();
        auto iB = tensor_iter_valonly<dt_int8>(src1).begin();
        auto pD = dst.ptr<dst_ctype>();
        for (size_t i = 0; i < size; i++) {
            *pD = elemwise_multi_type::round_shr_saturate<ctype, dst_ctype>(*iA, *iB);
            ++iA;
            ++iB;
            ++pD;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

template <typename ctype>
void ElemwiseMultiTypeImpl::dispatch_fuse_add_rmulh_round_shr_saturate(
        const ElemwiseOpParamN<6>& param, const TensorND& dst) {
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto src3 = param[3];
    auto src4 = param[4];
    auto src5 = param[5];
    auto work = [size, src0, src1, src2, src3, src4, src5, dst]() {
        auto i0 = tensor_iter_valonly<ctype>(src0).begin();
        auto i1 = tensor_iter_valonly<ctype>(src1).begin();
        auto i2 = tensor_iter_valonly<ctype>(src2).begin();
        auto ioff = tensor_iter_valonly<dt_int8>(src3).begin();
        auto imin = tensor_iter_valonly<dt_int8>(src4).begin();
        auto imax = tensor_iter_valonly<dt_int8>(src5).begin();
        auto dst_ptr = dst.ptr<dt_int8>();
        for (size_t i = 0; i < size; ++i) {
            auto res = elemwise_multi_type::round_shr_saturate<ctype, dt_int8>(
                    round_mulh_saturate<ctype>(*i0 + *i1, *i2), *ioff);
            res = std::min(res, *imax);
            res = std::max(res, *imin);
            dst_ptr[i] = res;
            ++i0;
            ++i1;
            ++i2;
            ++ioff;
            ++imin;
            ++imax;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
        const ElemwiseOpParamN<6>& param, const TensorND& dst) {
    dispatch_fuse_add_rmulh_round_shr_saturate<dt_int16>(param, dst);
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
        const ElemwiseOpParamN<6>& param, const TensorND& dst) {
    dispatch_fuse_add_rmulh_round_shr_saturate<dt_int32>(param, dst);
}

void ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi16(
        const ElemwiseOpParamN<2>& param, const TensorND& dst) {
    switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                                        \
    case DTypeTrait<t>::enumv:                                                       \
        return dispatch_round_shr_saturate_iXxi8xiX<DTypeTrait<t>::ctype, dt_int16>( \
                param, dst);
        cb(::megdnn::dtype::Int32);
        cb(::megdnn::dtype::Int16);
#undef cb
        default:
            megdnn_throw("unsupported src dtype");
    }
}

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
