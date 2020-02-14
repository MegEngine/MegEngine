/**
 * \file dnn/src/cuda/elemwise_multi_type/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/tensor_iter.h"

#include "src/common/elemwise/each_mode.inl"
#include "src/cuda/elemwise_multi_type/kern.cuh"
#include "src/cuda/elemwise_multi_type/kern_ops.cuh"
#include "src/cuda/elemwise_multi_type/opr_impl.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_int16x32x32x32(
        const ElemwiseOpParamN<3>& param, dt_int32* dst) {
    BroadcastChannelInfo binfo0, binfo1;
    if (is_vector(param[0].layout) &&
        is_broadcasted_channel_like(param[1].layout, binfo0) &&
        is_broadcasted_channel_like(param[2].layout, binfo1) &&
        binfo0 == binfo1) {
        elemwise_multi_type::fma3_int16x32x32x32_1c1(
                param, dst, cuda_stream(this->handle()));
        return;
    }
    megdnn_throw("unsupported fma3 int16x32x32x32 layout");
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_iXxf32xf32xi8(
        const ElemwiseOpParamN<3>& param, dt_int8* dst) {
    Broadcast1xInfo binfo0, binfo1;
    auto p1 = param[1].ptr<float>(), p2 = param[2].ptr<float>();
    auto stream = cuda_stream(this->handle());
    if (is_vector(param[0].layout) &&
        is_broadcasted_1x(param[1].layout, binfo0) &&
        is_broadcasted_1x(param[2].layout, binfo1) && binfo0 == binfo1) {
        switch (param[0].layout.dtype.enumv()) {
#define cb(t)                                                                \
    case DTypeTrait<t>::enumv:                                               \
        elemwise_multi_type::fma3_iXxf32xf32xi8_bcast_1x(                    \
                param[0].ptr<DTypeTrait<t>::ctype>(), p1, p2, dst, binfo0.x, \
                binfo0.y, stream);                                           \
        return;
            MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
            default:
                megdnn_throw("bad dtype");
        }
        return;
    }
    megdnn_throw("unsupported fma3 iXxf32xf32xi8 layout");
}

void ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi8(
        const ElemwiseOpParamN<2>& param, dt_int8* dst) {
    auto stream = cuda_stream(this->handle());
    if (is_vector(param[0].layout) && is_broadcasted_scalar(param[1].layout)) {
        switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(t)                                                 \
    case DTypeTrait<t>::enumv:                                      \
        elemwise_multi_type::round_shr_saturate_iXxi8xiX_scalar<    \
                DTypeTrait<t>::ctype, dt_int8>(param, dst, stream); \
        return;
            DISPATCH(::megdnn::dtype::Int32)
            DISPATCH(::megdnn::dtype::Int16)
            DISPATCH(::megdnn::dtype::Int8)
#undef DISPATCH
            default:
                megdnn_throw(
                        "Unsupported data type for ElemwiseMultiType "
                        "(Mode=ROUND_SHR_SATURATE_IXxI8xI8): need an integer "
                        "tensor");
        }
    }
    megdnn_throw(
            "Unsupported input layout for ElemwiseMultiType "
            "(Mode=ROUND_SHR_SATURATE_IXxI8xI8): need a contiguous src[0] and "
            "a scalar src[1]");
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
        const ElemwiseOpParamN<6>& param, dt_int8* dst) {
    auto stream = cuda_stream(this->handle());
    BroadcastChannelInfo info;
    if (is_vector(param[0].layout) &&
        is_broadcasted_channel_like(param[1].layout, info) &&
        is_broadcasted_scalar(param[2].layout) &&
        is_broadcasted_scalar(param[3].layout) &&
        is_broadcasted_scalar(param[4].layout) &&
        is_broadcasted_scalar(param[5].layout)) {
        elemwise_multi_type::fuse_add_rmulh_round_shr_saturate_bcast_1c11<
                dt_int16>(param, dst, stream);
        return;
    }
    megdnn_throw(
            "Unsupported input layout for ElemwiseMultiType "
            "(Mode=FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8): the first "
            "and the second input should be contiguous, the others should be "
            "scalar.");
}

void ElemwiseMultiTypeImpl::on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
        const ElemwiseOpParamN<6>& param, dt_int8* dst) {
    auto stream = cuda_stream(this->handle());
    BroadcastChannelInfo info;
    if (is_vector(param[0].layout) &&
        is_broadcasted_channel_like(param[1].layout, info) &&
        is_broadcasted_scalar(param[2].layout) &&
        is_broadcasted_scalar(param[3].layout) &&
        is_broadcasted_scalar(param[4].layout) &&
        is_broadcasted_scalar(param[5].layout)) {
        elemwise_multi_type::fuse_add_rmulh_round_shr_saturate_bcast_1c11<
                dt_int32>(param, dst, stream);
        return;
    }
    megdnn_throw(
            "Unsupported input layout for ElemwiseMultiType "
            "(Mode=FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8): the first "
            "and the second input should be contiguous, the others should be "
            "scalar.");
}

void ElemwiseMultiTypeImpl::on_round_shr_saturate_iXxi8xi16(
        const ElemwiseOpParamN<2>& param, dt_int16* dst) {
    auto stream = cuda_stream(this->handle());
    if (is_vector(param[0].layout) && is_broadcasted_scalar(param[1].layout)) {
        switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(t)                                                  \
    case DTypeTrait<t>::enumv:                                       \
        elemwise_multi_type::round_shr_saturate_iXxi8xiX_scalar<     \
                DTypeTrait<t>::ctype, dt_int16>(param, dst, stream); \
        return;
            DISPATCH(::megdnn::dtype::Int32)
            DISPATCH(::megdnn::dtype::Int16)
#undef DISPATCH
            default:
                megdnn_throw(
                        "Unsupported data type for ElemwiseMultiType "
                        "(Mode=ROUND_SHR_SATURATE_IXxI8xI8): need an integer "
                        "tensor");
        }
    }
    megdnn_throw(
            "Unsupported input layout for ElemwiseMultiType "
            "(Mode=ROUND_SHR_SATURATE_IXxI8xI8): need a contiguous src[0] and "
            "a scalar src[1]");
}

namespace {

template <int arity, typename _src_ctype, typename _dst_ctype>
struct ModeDispatcher;

#define _cb_dispatch_mode(_m)                                             \
    case param::Elemwise::Mode::_m:                                       \
        do {                                                              \
            using KernImpl =                                              \
                    ElemwiseKern<megcorePlatformCUDA,                     \
                                 param_enumv::Elemwise::Mode::_m, float>; \
            using Op = kern_ops_quantized::QuantizedMultiTypeOp<          \
                    arity, src_ctype, dst_ctype, KernImpl>;               \
            Op op(src_params, dst, dst_param);                            \
            return run_elemwise<Op, src_ctype, arity>(param, stream, op); \
        } while (0);

#define IMPL_MODE_DISPATCHER(_arity, _src_ctype, _dst_ctype)               \
    template <>                                                            \
    struct ModeDispatcher<_arity, _src_ctype, _dst_ctype> {                \
        static constexpr int arity = _arity;                               \
        using src_ctype = _src_ctype;                                      \
        using dst_ctype = _dst_ctype;                                      \
        static void run(                                                   \
                const ElemwiseOpParamN<_arity>& param, _dst_ctype* dst,    \
                const SmallVector<CudaDTypeParam<_src_ctype>>& src_params, \
                const CudaDTypeParam<_dst_ctype>& dst_param,               \
                param::Elemwise::Mode mode, cudaStream_t stream) {         \
            megdnn_assert(src_params.size() == _arity);                    \
            switch (mode) {                                                \
                FOREACH(_cb_dispatch_mode)                                 \
                default:                                                   \
                    megdnn_throw("bad mode");                              \
            }                                                              \
        }                                                                  \
    }

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_FLOAT
IMPL_MODE_DISPATCHER(1, dt_qint8, dt_qint8);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_FLOAT
IMPL_MODE_DISPATCHER(2, dt_qint8, dt_qint8);
#undef FOREACH

#define FOREACH MEGDNN_FOREACH_ELEMWISE_MODE_TERNARY_FLOAT
IMPL_MODE_DISPATCHER(3, dt_qint8, dt_qint8);
#undef FOREACH

#define FOREACH(cb)                            \
    MEGDNN_ELEMWISE_MODE_ENABLE(RELU, cb)      \
    MEGDNN_ELEMWISE_MODE_ENABLE(SIGMOID, cb)   \
    MEGDNN_ELEMWISE_MODE_ENABLE(TANH, cb)      \
    MEGDNN_ELEMWISE_MODE_ENABLE(FAST_TANH, cb) \
    MEGDNN_ELEMWISE_MODE_ENABLE(H_SWISH, cb)
IMPL_MODE_DISPATCHER(1, dt_qint8, dt_qint32);
IMPL_MODE_DISPATCHER(1, dt_qint32, dt_qint8);
#undef FOREACH

#define FOREACH(cb)                                   \
    MEGDNN_ELEMWISE_MODE_ENABLE(ADD, cb)              \
    MEGDNN_ELEMWISE_MODE_ENABLE(FUSE_ADD_RELU, cb)    \
    MEGDNN_ELEMWISE_MODE_ENABLE(FUSE_ADD_SIGMOID, cb) \
    MEGDNN_ELEMWISE_MODE_ENABLE(FUSE_ADD_TANH, cb)    \
    MEGDNN_ELEMWISE_MODE_ENABLE(FUSE_ADD_H_SWISH, cb)
IMPL_MODE_DISPATCHER(2, dt_qint8, dt_qint32);
IMPL_MODE_DISPATCHER(2, dt_qint32, dt_qint8);
#undef FOREACH

#undef _cb_dispatch_mode
#undef IMPL_MODE_DISPATCHER

template <typename ctype_src>
void dispatch_src_ctype(const ElemwiseOpParamN<1>&, const TensorND& dst_tensor,
                        Elemwise::Mode, cudaStream_t);

#define DISPATCH(_dt)                                                       \
    case DTypeTrait<_dt>::enumv: {                                          \
        auto param_a = param[0].layout.dtype.param<ctype_src>();            \
        auto dst_param = dst_tensor.layout.dtype.param<_dt>();              \
        ModeDispatcher<1, ctype_src, typename DTypeTrait<_dt>::ctype>::run( \
                param, dst_tensor.ptr<typename DTypeTrait<_dt>::ctype>(),   \
                {param_a}, dst_param, mode, stream);                        \
        break;                                                              \
    }

template <>
void dispatch_src_ctype<dt_qint8>(const ElemwiseOpParamN<1>& param,
                                  const TensorND& dst_tensor,
                                  Elemwise::Mode mode, cudaStream_t stream) {
    typedef dt_qint8 ctype_src;
    switch (dst_tensor.layout.dtype.enumv()) {
        DISPATCH(dtype::QuantizedS8);
        DISPATCH(dtype::QuantizedS32);
        default:
            megdnn_throw(ssprintf(
                    "Unsupported output dtype %s for ElemwiseMultiType",
                    dst_tensor.layout.dtype.name()));
    }
}

template <>
void dispatch_src_ctype<dt_qint32>(const ElemwiseOpParamN<1>& param,
                                   const TensorND& dst_tensor,
                                   Elemwise::Mode mode, cudaStream_t stream) {
    typedef dt_qint32 ctype_src;
    switch (dst_tensor.layout.dtype.enumv()) {
        DISPATCH(dtype::QuantizedS8);
        default:
            megdnn_throw(ssprintf(
                    "Unsupported output dtype %s for ElemwiseMultiType",
                    dst_tensor.layout.dtype.name()));
    }
}

#undef DISPATCH

#define DISPATCH(_dt)                                                       \
    case DTypeTrait<_dt>::enumv: {                                          \
        auto param_a = param[0].layout.dtype.param<ctype_src>();            \
        auto param_b = param[1].layout.dtype.param<ctype_src>();            \
        auto dst_param = dst_tensor.layout.dtype.param<_dt>();              \
        ModeDispatcher<2, ctype_src, typename DTypeTrait<_dt>::ctype>::run( \
                param, dst_tensor.ptr<typename DTypeTrait<_dt>::ctype>(),   \
                {param_a, param_b}, dst_param, mode, stream);               \
        break;                                                              \
    }

template <typename ctype_src>
void dispatch_src_ctype(const ElemwiseOpParamN<2>& param,
                        const TensorND& dst_tensor, Elemwise::Mode mode,
                        cudaStream_t stream);
template <>
void dispatch_src_ctype<dt_qint8>(const ElemwiseOpParamN<2>& param,
                                  const TensorND& dst_tensor,
                                  Elemwise::Mode mode, cudaStream_t stream) {
    typedef dt_qint8 ctype_src;
    switch (dst_tensor.layout.dtype.enumv()) {
        DISPATCH(dtype::QuantizedS8);
        DISPATCH(dtype::QuantizedS32);
        default:
            megdnn_throw(ssprintf(
                    "Unsupported output dtype %s for ElemwiseMultiType",
                    dst_tensor.layout.dtype.name()));
    }
}

template <>
void dispatch_src_ctype<dt_qint32>(const ElemwiseOpParamN<2>& param,
                                   const TensorND& dst_tensor,
                                   Elemwise::Mode mode, cudaStream_t stream) {
    typedef dt_qint32 ctype_src;
    switch (dst_tensor.layout.dtype.enumv()) {
        DISPATCH(dtype::QuantizedS8);
        default:
            megdnn_throw(ssprintf(
                    "Unsupported output dtype %s for ElemwiseMultiType",
                    dst_tensor.layout.dtype.name()));
    }
}
#undef DISPATCH

#define DISPATCH(_dt)                                                       \
    case DTypeTrait<_dt>::enumv: {                                          \
        auto param_a = param[0].layout.dtype.param<ctype_src>();            \
        auto param_b = param[1].layout.dtype.param<ctype_src>();            \
        auto param_c = param[2].layout.dtype.param<ctype_src>();            \
        auto dst_param = dst_tensor.layout.dtype.param<_dt>();              \
        ModeDispatcher<3, ctype_src, typename DTypeTrait<_dt>::ctype>::run( \
                param, dst_tensor.ptr<typename DTypeTrait<_dt>::ctype>(),   \
                {param_a, param_b, param_c}, dst_param, mode, stream);      \
        break;                                                              \
    }

template <typename ctype_src>
void dispatch_src_ctype(const ElemwiseOpParamN<3>& param,
                        const TensorND& dst_tensor, Elemwise::Mode mode,
                        cudaStream_t stream);
template <>
void dispatch_src_ctype<dt_qint8>(const ElemwiseOpParamN<3>& param,
                                  const TensorND& dst_tensor,
                                  Elemwise::Mode mode, cudaStream_t stream) {
    typedef dt_qint8 ctype_src;
    switch (dst_tensor.layout.dtype.enumv()) {
        DISPATCH(dtype::QuantizedS8);
        default:
            megdnn_throw(ssprintf(
                    "Unsupported output dtype %s for ElemwiseMultiType",
                    dst_tensor.layout.dtype.name()));
    }
}

#undef DISPATCH

}  // namespace

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<1>& param,
                                              const TensorND& dst_tensor,
                                              Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                    param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32,
            "expect inputs dtype to be qint8/qint32, but got: %s",
            param[0].layout.dtype.name());
    auto stream = cuda_stream(this->handle());
    switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(_dt)                                                          \
    case DTypeTrait<_dt>::enumv: {                                             \
        dispatch_src_ctype<typename DTypeTrait<_dt>::ctype>(param, dst_tensor, \
                                                            mode, stream);     \
        break;                                                                 \
    }

        DISPATCH(dtype::QuantizedS8);
        DISPATCH(dtype::QuantizedS32);

        default:
            megdnn_throw(
                    ssprintf("Unsupported input dtype %s for ElemwiseMultiType",
                             param[0].layout.dtype.name()));
    }

#undef DISPATCH
}

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<2>& param,
                                              const TensorND& dst_tensor,
                                              Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.enumv() ==
                  param[1].layout.dtype.enumv());
    megdnn_assert(
            param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                    param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS32,
            "expect inputs dtype to be qint8/qint32, but got: %s",
            param[0].layout.dtype.name());
    auto stream = cuda_stream(this->handle());
    switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(_dt)                                                          \
    case DTypeTrait<_dt>::enumv: {                                             \
        dispatch_src_ctype<typename DTypeTrait<_dt>::ctype>(param, dst_tensor, \
                                                            mode, stream);     \
        break;                                                                 \
    }

        DISPATCH(dtype::QuantizedS8);
        DISPATCH(dtype::QuantizedS32);

        default:
            megdnn_throw(
                    ssprintf("Unsupported input dtype %s for ElemwiseMultiType",
                             param[0].layout.dtype.name()));
    }

#undef DISPATCH
}

void ElemwiseMultiTypeImpl::on_quantized_mode(const ElemwiseOpParamN<3>& param,
                                              const TensorND& dst_tensor,
                                              Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.enumv() ==
                  param[1].layout.dtype.enumv());
    megdnn_assert(param[0].layout.dtype.enumv() ==
                  param[2].layout.dtype.enumv());

    megdnn_assert(
            param[0].layout.dtype.enumv() == DTypeEnum::QuantizedS8,
            "expect inputs dtype to be qint8, but got: %s",
            param[0].layout.dtype.name());
    auto stream = cuda_stream(this->handle());
    switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(_dt)                                                          \
    case DTypeTrait<_dt>::enumv: {                                             \
        dispatch_src_ctype<typename DTypeTrait<_dt>::ctype>(param, dst_tensor, \
                                                            mode, stream);     \
        break;                                                                 \
    }

        DISPATCH(dtype::QuantizedS8);

        default:
            megdnn_throw(
                    ssprintf("Unsupported input dtype %s for ElemwiseMultiType",
                             param[0].layout.dtype.name()));
    }

#undef DISPATCH
}

// vim: syntax=cpp.doxygen
