/**
 * \file dnn/src/fallback/elemwise_multi_type/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise_multi_type/kern_defs.cuh"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace fallback;

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_int16x32x32x32(
        const ElemwiseOpParamN<3>& param, dt_int32* dst) {
    BroadcastChannelInfo binfo0, binfo1;
    if (is_vector(param[0].layout) &&
        is_broadcasted_channel_like(param[1].layout, binfo0) &&
        is_broadcasted_channel_like(param[2].layout, binfo1) &&
        binfo0 == binfo1) {
        auto pa = param[0].ptr<dt_int16>();
        auto pb = param[1].ptr<dt_int32>();
        auto pc = param[2].ptr<dt_int32>();
        auto x = binfo0.x, y = binfo0.y, z = binfo0.z;
        auto work = [pa, pb, pc, dst, x, y, z]() {
            const dt_int16* __restrict__ a = pa;
            const dt_int32* __restrict__ b = pb;
            const dt_int32* __restrict__ c = pc;
            dt_int32* __restrict__ d = dst;
            for (size_t j = 0; j < y; ++j) {
                auto bv = b[j], cv = c[j];
                for (size_t i = 0; i < x; ++i) {
                    auto off = i * (y * z) + j * z, offt = off + z;
                    for (; off + 4 <= offt; off += 4) {
                        d[off + 0] = a[off + 0] * bv + cv;
                        d[off + 1] = a[off + 1] * bv + cv;
                        d[off + 2] = a[off + 2] * bv + cv;
                        d[off + 3] = a[off + 3] * bv + cv;
                    }
                    for (; off < offt; ++off) {
                        d[off] = a[off] * bv + cv;
                    }
                }
            }
        };

        MEGDNN_DISPATCH_CPU_KERN_OPR(work());
        return;
    }

    naive::ElemwiseMultiTypeImpl::on_fuse_mul_add3_int16x32x32x32(param, dst);
}

template <typename ctype>
void ElemwiseMultiTypeImpl::dispatch_fma3_iXxf32xf32xi8_bcast_1x(
        const ElemwiseOpParamN<3>& param, const Broadcast1xInfo& binfo,
        dt_int8* dst) {
    auto pa = param[0].ptr<ctype>();
    auto pb = param[1].ptr<dt_float32>();
    auto pc = param[2].ptr<dt_float32>();
    size_t x = binfo.x, y = binfo.y;
    auto work = [pa, pb, pc, dst, x, y]() {
        elemwise_multi_type::Fma3iXxf32xf32xiYOp<ctype, dt_int8> op;
        const ctype* __restrict__ a = pa;
        const dt_float32* __restrict__ b = pb;
        const dt_float32* __restrict__ c = pc;
        dt_int8* __restrict__ d = dst;
        for (size_t i = 0; i < x; ++i) {
            size_t j = 0;
            for (; j + 4 <= y; j += 4) {
                d[j + 0] = op(a[j + 0], b[j + 0], c[j + 0]);
                d[j + 1] = op(a[j + 1], b[j + 1], c[j + 1]);
                d[j + 2] = op(a[j + 2], b[j + 2], c[j + 2]);
                d[j + 3] = op(a[j + 3], b[j + 3], c[j + 3]);
            }
            for (; j < y; ++j) {
                d[j] = op(a[j], b[j], c[j]);
            }

            d += y;
            a += y;
        }
    };

    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_iXxf32xf32xi8(
        const ElemwiseOpParamN<3>& param, dt_int8* dst) {
    Broadcast1xInfo binfo0, binfo1;
    if (is_vector(param[0].layout) &&
        is_broadcasted_1x(param[1].layout, binfo0) &&
        is_broadcasted_1x(param[2].layout, binfo1) && binfo0 == binfo1) {
        switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                              \
    case DTypeTrait<t>::enumv:                                             \
        return dispatch_fma3_iXxf32xf32xi8_bcast_1x<DTypeTrait<t>::ctype>( \
                param, binfo0, dst);
            MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
            default:
                megdnn_throw("unsupported src dtype");
        }
    }

    // fallback to naive
    switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                             \
    case DTypeTrait<t>::enumv:                                            \
        return naive::ElemwiseMultiTypeImpl::dispatch_fma3_iXxf32xf32xi8< \
                DTypeTrait<t>::ctype>(param, dst);
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
        default:
            megdnn_throw("unsupported src dtype");
    }
}

template <typename ctype, typename dst_ctype>
void ElemwiseMultiTypeImpl::dispatch_round_shr_saturate_iXxi8xiX_bcast_scalar(
        const ElemwiseOpParamN<2>& param, dst_ctype* dst) {
    auto x_ptr = param[0].ptr<ctype>();
    auto k = param[1].ptr<dt_int8>()[0];
    size_t size = param.size;
    auto work = [x_ptr, k, size, dst]() {
        const ctype* __restrict__ xp = x_ptr;
        dst_ctype* __restrict__ dp = dst;
        for (size_t i = 0; i < size; i++) {
            dp[i] = elemwise_multi_type::round_shr_saturate<ctype, dst_ctype>(
                    xp[i], k);
        }
    };

    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi8(
        const ElemwiseOpParamN<2>& param, megdnn::dt_int8* dst) {
    if (is_vector(param[0].layout) && is_broadcasted_scalar(param[1].layout)) {
        switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                     \
    case DTypeTrait<t>::enumv:                                    \
        return dispatch_round_shr_saturate_iXxi8xiX_bcast_scalar< \
                DTypeTrait<t>::ctype, dt_int8>(param, dst);
            MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
            default:
                megdnn_throw(
                        "ElemwiseMultiType: unsupported src dtype for "
                        "ROUND_SHR_SATURATE");
        }
    }

    naive::ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi8(param, dst);
}

void ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi16(
        const ElemwiseOpParamN<2>& param, megdnn::dt_int16* dst) {
    if (is_vector(param[0].layout) && is_broadcasted_scalar(param[1].layout)) {
        switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                     \
    case DTypeTrait<t>::enumv:                                    \
        return dispatch_round_shr_saturate_iXxi8xiX_bcast_scalar< \
                DTypeTrait<t>::ctype, dt_int16>(param, dst);
            cb(::megdnn::dtype::Int32);
            cb(::megdnn::dtype::Int16);
#undef cb
            default:
                megdnn_throw(
                        "ElemwiseMultiType: unsupported src dtype for "
                        "ROUND_SHR_SATURATE");
        }
    }

    naive::ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi16(param, dst);
}

template <typename ctype>
void ElemwiseMultiTypeImpl::
        dispatch_fuse_add_rmulh_round_shr_saturate_bcast_1c11(
                const ElemwiseOpParamN<6>& param, megdnn::dt_int8* dst,
                const BroadcastChannelInfo& broadcast_info) {
    auto work = [param, dst, broadcast_info]() {
        auto x_ptr = param[0].ptr<ctype>();
        auto b_ptr = param[1].ptr<ctype>();
        auto M = param[2].ptr<ctype>()[0];
        auto k = param[3].ptr<dt_int8>()[0];
        auto minv = param[4].ptr<dt_int8>()[0];
        auto maxv = param[5].ptr<dt_int8>()[0];
        auto dst_ptr = dst;
        auto batch_stride = broadcast_info.y * broadcast_info.z;
        auto channel_stride = broadcast_info.z;
        for (size_t n = 0; n < broadcast_info.x; n++) {
            const ctype* __restrict__ xp = x_ptr;
            dt_int8* __restrict__ dp = dst_ptr;
            for (size_t chan = 0; chan < broadcast_info.y; chan++) {
                const ctype bias = b_ptr[chan * param[1].layout.stride[1]];
                for (size_t i = 0; i < broadcast_info.z; i++) {
                    auto res = elemwise_multi_type::round_shr_saturate<ctype,
                                                                       dt_int8>(
                            round_mulh_saturate<ctype>(xp[i] + bias, M), k);
                    res = std::min(res, maxv);
                    res = std::max(res, minv);
                    dp[i] = res;
                }
                xp += channel_stride;
                dp += channel_stride;
            }
            x_ptr += batch_stride;
            dst_ptr += batch_stride;
        }
    };

    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
        const ElemwiseOpParamN<6>& param, megdnn::dt_int8* dst) {
    bool all_scalar = true;
    for (int i = 3; i < 6; i++) {
        all_scalar &= is_broadcasted_scalar(param[i].layout);
    }
    BroadcastChannelInfo info;
    if (!all_scalar || !is_broadcasted_channel_like(param[1].layout, info)) {
        return naive::ElemwiseMultiTypeImpl::
                on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(param, dst);
    }
    dispatch_fuse_add_rmulh_round_shr_saturate_bcast_1c11<dt_int16>(param, dst,
                                                                    info);
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
        const ElemwiseOpParamN<6>& param, megdnn::dt_int8* dst) {
    bool all_scalar = true;
    for (int i = 3; i < 6; i++) {
        all_scalar &= is_broadcasted_scalar(param[i].layout);
    }
    BroadcastChannelInfo info;
    if (!all_scalar || !is_broadcasted_channel_like(param[1].layout, info)) {
        return naive::ElemwiseMultiTypeImpl::
                on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(param, dst);
    }
    dispatch_fuse_add_rmulh_round_shr_saturate_bcast_1c11<dt_int32>(param, dst,
                                                                    info);
}

// vim: syntax=cpp.doxygen
