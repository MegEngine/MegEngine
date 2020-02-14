/**
 * \file dnn/src/cuda/elemwise_multi_type/kern_ops.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/elemwise_multi_type/kern.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
using namespace elemwise_intl;

namespace kern_ops {

//! a * b + c, where a is [x, y, z] and b, c both [1, y, 1]
struct Fma3Int16x32x32x32Bcast101Op {
    ParamElemVisitor<1, dt_int16, BCAST_OTHER> a;
    ParamElemVisitor<3, dt_int32, BCAST_101> b, c;

    dt_int32* dst;

#if MEGDNN_CC_CUDA
    __device__ __forceinline__ void thread_init(uint32_t idx) {
        a.thread_init(idx);
        b.thread_init(idx);
        c.thread_init(idx);
    }

    __device__ __forceinline__ void on(uint32_t idx) {
        dst[idx] = a.at(idx) * b.at(idx) + c.at(idx);
    }

    __device__ __forceinline__ void next() {
        a.next();
        b.next();
        c.next();
    }
#endif
};

template <typename stype, typename dst_type>
struct RoundShrSaturateIXxBcastScalarOp {
    ParamElemVisitor<1, stype, BCAST_OTHER> a;
    ParamElemVisitor<1, dt_int8, BCAST_FULL> b;

    dst_type* dst;

#if MEGDNN_CC_CUDA
    __device__ __forceinline__ void thread_init(uint32_t idx) {
        a.thread_init(idx);
        b.thread_init(idx);
    }

    __device__ __forceinline__ void on(uint32_t idx) {
        stype result =
                rounding_shift_right_away_from_zero(a.at(idx), b.at(idx));
        result = result < INT8_MAX ? result : INT8_MAX;
        result = result > INT8_MIN ? result : INT8_MIN;
        dst[idx] = static_cast<dst_type>(result);
    }

    __device__ __forceinline__ void next() {
        a.next();
        b.next();
    }
#endif
};

template <typename stype>
struct FuseAddRmulhRoundingShrBcastScalarOp {
    ParamElemVisitor<1, stype, BCAST_OTHER> x;
    ParamElemVisitor<3, stype, BCAST_101> b;
    ParamElemVisitor<1, stype, BCAST_FULL> M;
    ParamElemVisitor<1, dt_int8, BCAST_FULL> k;
    ParamElemVisitor<1, dt_int8, BCAST_FULL> minv;
    ParamElemVisitor<1, dt_int8, BCAST_FULL> maxv;

    dt_int8* dst;

#if MEGDNN_CC_CUDA
    __device__ __forceinline__ void thread_init(uint32_t idx) {
        x.thread_init(idx);
        b.thread_init(idx);
        M.thread_init(idx);
        k.thread_init(idx);
        minv.thread_init(idx);
        maxv.thread_init(idx);
    }

    __device__ __forceinline__ void on(uint32_t idx) {
        stype result = rounding_shift_right_away_from_zero(
                round_mulh_saturate<stype>(x.at(idx) + b.at(idx), M.at(idx)),
                k.at(idx));
        stype lminv = minv.at(idx);
        stype lmaxv = maxv.at(idx);
        result = lminv < result ? result : lminv;
        result = result < lmaxv ? result : lmaxv;
        dst[idx] = static_cast<dt_int8>(result);
    }

    __device__ __forceinline__ void next() {
        x.next();
        b.next();
    }
#endif
};
}  // namespace kern_ops

#ifndef MEGDNN_ELEMWISE_MODE_ENABLE
#define MEGDNN_ELEMWISE_MODE_ENABLE(_mode, _cb) _cb(_mode)
#endif

namespace kern_ops_quantized {

template <int arity, typename ctype_src, typename ctype_dst, typename KernImpl,
          typename enable = void>
struct QuantizedMultiTypeOp;

template <typename ctype_src, typename ctype_dst, typename KernImpl>
struct QuantizedMultiTypeOp<
        1, ctype_src, ctype_dst, KernImpl,
        typename std::enable_if<
                std::is_same<ctype_src, dt_qint8>::value ||
                std::is_same<ctype_src, dt_qint32>::value ||
                std::is_same<ctype_src, dt_quint8>::value>::type> {
    ctype_dst* dst;
    CudaDTypeParam<ctype_dst> dst_param;
    CudaDTypeParam<ctype_src> param_a;
    typedef typename elemwise_intl::VectTypeTrait<ctype_src>::vect_type
            src_vect_type;
    typedef typename elemwise_intl::VectTypeTrait<ctype_dst>::vect_type
            dst_vect_type;

#if !MEGDNN_CC_CUDA
    QuantizedMultiTypeOp(
            const SmallVector<CudaDTypeParam<ctype_src>>& src_params,
            ctype_dst* dst, const CudaDTypeParam<ctype_dst>& dst_param)
            : dst{dst}, dst_param{dst_param} {
        param_a = src_params[0];
    }
#endif

#if MEGDNN_CC_CUDA
    __device__ __forceinline__ ctype_dst apply(ctype_src v1) {
        float fv1 = param_a.dequantize(v1);
        float rv = KernImpl::apply(fv1);
        return dst_param.quantize(rv);
    }

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src a) {
        dst[idx] = dst_param.quantize(KernImpl::apply(param_a.dequantize(a)));
    }

    __device__ __forceinline__ void operator()(uint32_t idx, src_vect_type a) {
        ctype_src a_x(a.x), a_y(a.y), a_z(a.z), a_w(a.w);
        ctype_dst x = apply(a_x), y = apply(a_y), z = apply(a_z),
                  w = apply(a_w);
        *(dst_vect_type*)(&dst[idx]) =
                elemwise_intl::VectTypeTrait<ctype_dst>::make_vector(x, y, z,
                                                                     w);
    }
#endif
};

template <typename ctype_src, typename ctype_dst, typename KernImpl>
struct QuantizedMultiTypeOp<
        2, ctype_src, ctype_dst, KernImpl,
        typename std::enable_if<
                std::is_same<ctype_src, dt_qint8>::value ||
                std::is_same<ctype_src, dt_qint32>::value ||
                std::is_same<ctype_src, dt_quint8>::value>::type> {
    ctype_dst* dst;
    CudaDTypeParam<ctype_dst> dst_param;
    CudaDTypeParam<ctype_src> param_a, param_b;
    typedef typename elemwise_intl::VectTypeTrait<ctype_src>::vect_type
            src_vect_type;
    typedef typename elemwise_intl::VectTypeTrait<ctype_dst>::vect_type
            dst_vect_type;

#if !MEGDNN_CC_CUDA
    QuantizedMultiTypeOp(
            const SmallVector<CudaDTypeParam<ctype_src>>& src_params,
            ctype_dst* dst, const CudaDTypeParam<ctype_dst>& dst_param)
            : dst{dst}, dst_param{dst_param} {
        param_a = src_params[0];
        param_b = src_params[1];
    }
#endif

#if MEGDNN_CC_CUDA
    __device__ __forceinline__ ctype_dst apply(ctype_src v1, ctype_src v2) {
        float fv1 = param_a.dequantize(v1), fv2 = param_b.dequantize(v2);
        float rv = KernImpl::apply(fv1, fv2);
        return dst_param.quantize(rv);
    }

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src a,
                                               ctype_src b) {
        dst[idx] = dst_param.quantize(
                KernImpl::apply(param_a.dequantize(a), param_b.dequantize(b)));
    }

    __device__ __forceinline__ void operator()(uint32_t idx, src_vect_type a,
                                               src_vect_type b) {
        ctype_src a_x(a.x), a_y(a.y), a_z(a.z), a_w(a.w), b_x(b.x), b_y(b.y),
                b_z(b.z), b_w(b.w);
        ctype_dst x = apply(a_x, b_x), y = apply(a_y, b_y), z = apply(a_z, b_z),
                  w = apply(a_w, b_w);
        *(dst_vect_type*)(&dst[idx]) =
                elemwise_intl::VectTypeTrait<ctype_dst>::make_vector(x, y, z,
                                                                     w);
    }
#endif
};

template <typename ctype_src, typename ctype_dst, typename KernImpl>
struct QuantizedMultiTypeOp<
        3, ctype_src, ctype_dst, KernImpl,
        typename std::enable_if<
                std::is_same<ctype_src, dt_qint8>::value ||
                std::is_same<ctype_src, dt_qint32>::value ||
                std::is_same<ctype_src, dt_quint8>::value>::type> {
    ctype_dst* dst;
    CudaDTypeParam<ctype_dst> dst_param;
    CudaDTypeParam<ctype_src> param_a, param_b, param_c;
    typedef typename elemwise_intl::VectTypeTrait<ctype_src>::vect_type
            src_vect_type;
    typedef typename elemwise_intl::VectTypeTrait<ctype_dst>::vect_type
            dst_vect_type;

#if !MEGDNN_CC_CUDA
    QuantizedMultiTypeOp(
            const SmallVector<CudaDTypeParam<ctype_src>>& src_params,
            ctype_dst* dst, const CudaDTypeParam<ctype_dst>& dst_param)
            : dst{dst}, dst_param{dst_param} {
        param_a = src_params[0];
        param_b = src_params[1];
        param_c = src_params[2];
    }
#endif

#if MEGDNN_CC_CUDA
    __device__ __forceinline__ ctype_dst apply(ctype_src v1, ctype_src v2,
                                               ctype_src v3) {
        float fv1 = param_a.dequantize(v1), fv2 = param_b.dequantize(v2),
              fv3 = param_c.dequantize(v3);
        float rv = KernImpl::apply(fv1, fv2, fv3);
        return dst_param.quantize(rv);
    }

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src a,
                                               ctype_src b, ctype_src c) {
        dst[idx] = dst_param.quantize(KernImpl::apply(param_a.dequantize(a),
                                                      param_b.dequantize(b),
                                                      param_c.dequantize(c)));
    }

    __device__ __forceinline__ void operator()(uint32_t idx, src_vect_type a,
                                               src_vect_type b,
                                               src_vect_type c) {
        ctype_src a_x(a.x), a_y(a.y), a_z(a.z), a_w(a.w), b_x(b.x), b_y(b.y),
                b_z(b.z), b_w(b.w), c_x(c.x), c_y(c.y), c_z(c.z), c_w(c.w);
        ctype_dst x = apply(a_x, b_x, c_x), y = apply(a_y, b_y, c_y),
                  z = apply(a_z, b_z, c_z), w = apply(a_w, b_w, c_w);
        *(dst_vect_type*)(&dst[idx]) =
                elemwise_intl::VectTypeTrait<ctype_dst>::make_vector(x, y, z,
                                                                     w);
    }
#endif
};

}  // namespace kern_ops_quantized

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
