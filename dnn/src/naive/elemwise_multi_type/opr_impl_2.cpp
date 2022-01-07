/**
 * \file dnn/src/naive/elemwise_multi_type/opr_impl_2.cpp
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

// vim: syntax=cpp.doxygen
