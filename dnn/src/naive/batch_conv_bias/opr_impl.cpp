/**
 * \file dnn/src/naive/batch_conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/batch_conv_bias/opr_impl.h"
#include "megdnn/oprs/nn.h"
#include "src/common/conv_bias.h"
#include "src/naive/conv_bias/opr_impl.h"
#include "src/naive/convolution/helper.h"

#include <cstring>
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;
using namespace convolution;

namespace {
struct BatchConvFilterVisitor {
    template <typename ftype>
    static ftype* get_current_ptr(ftype* fptr, size_t batch, size_t /* oc */,
                                  size_t /* oh */, size_t /* ow */,
                                  size_t filter_sizes) {
        return fptr + batch * filter_sizes;
    }
};
}  // namespace

WorkspaceBundle BatchConvBiasForwardImpl::get_workspace_bundle(
        dt_byte* raw_ptr, const TensorLayout& /* src */,
        const TensorLayout& /* flt */, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst) {
    size_t ws_bias_size = 0, ws_z_size = 0;
    if (bias.dtype.enumv() != dst.dtype.enumv()) {
        ws_z_size = TensorLayout{dst, bias.dtype}.span().dist_byte();
    }
    if (z.ndim > 0) {
        megdnn_assert(z.dtype.enumv() == DTypeEnum::QuantizedS8);
        megdnn_assert(z.eq_shape(dst));
        // (w * f + b).astype(float) + (z).astype(float)
        size_t f32_z_size =
                TensorLayout{z, dtype::Float32()}.span().dist_byte();
        ws_z_size = f32_z_size + f32_z_size;
    }
    return WorkspaceBundle{raw_ptr, {ws_bias_size, ws_z_size}};
}

size_t BatchConvBiasForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& flt,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst) {
    return get_workspace_bundle(nullptr, src, flt, bias, z, dst)
            .total_size_in_bytes();
}

void BatchConvBiasForwardImpl::exec(_megdnn_tensor_in src,
                                    _megdnn_tensor_in filter,
                                    _megdnn_tensor_in bias, _megdnn_tensor_in z,
                                    _megdnn_tensor_out dst,
                                    _megdnn_workspace workspace) {
    auto filter_meta = check_exec(src.layout, filter.layout, bias.layout,
                                  z.layout, dst.layout, workspace.size);
    WorkspaceBundle ws =
            get_workspace_bundle(workspace.raw_ptr, src.layout, filter.layout,
                                 bias.layout, z.layout, dst.layout);
    auto sfb = dst;
    if (bias.layout.dtype.enumv() != dst.layout.dtype.enumv()) {
        sfb = TensorND{ws.get(0), TensorLayout{dst.layout, bias.layout.dtype}};
    }

#define DISPATCH_RAW(in_dt, bias_dt, out_dt, cmode, func)                      \
    else if (src.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv &&    \
             filter.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv && \
             bias.layout.dtype.enumv() == DTypeTrait<dtype::bias_dt>::enumv && \
             sfb.layout.dtype.enumv() == DTypeTrait<dtype::out_dt>::enumv &&   \
             param().compute_mode == Param::ComputeMode::cmode) {              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                func(src, filter, bias, sfb, nullptr, filter_meta));           \
    }
#define DISPATCH(in_dt, out_dt)                                           \
    DISPATCH_RAW(in_dt, out_dt, out_dt, DEFAULT,                          \
                 (forward_bias<DTypeTrait<dtype::in_dt>::ctype,           \
                               DTypeTrait<dtype::in_dt>::ctype,           \
                               DTypeTrait<dtype::out_dt>::ctype,          \
                               DTypeTrait<dtype::out_dt>::ctype,          \
                               BatchConvBiasForward::CanonizedFilterMeta, \
                               BatchConvFilterVisitor>))
    if (0) {
    }
    DISPATCH(QuantizedS8, QuantizedS32)
    else {
        megdnn_throw(ssprintf(
                "unsupported naive BatchConvBias(%s, %s, %s, %s) -> %s",
                src.layout.dtype.name(), filter.layout.dtype.name(),
                bias.layout.dtype.name(), z.layout.dtype.name(),
                dst.layout.dtype.name()));
    }
#undef DISPATCH
#undef DISPATCH_RAW
    MEGDNN_DISPATCH_CPU_KERN_OPR(handle_z_inp_and_activation_naive(
            param().nonlineMode, sfb, z, dst,
            reinterpret_cast<dt_byte*>(ws.get(1))));
}

std::vector<BatchConvBiasForward::Algorithm*>
BatchConvBiasForwardImpl::get_all_algorithms(const TensorLayout&,
                                             const TensorLayout&,
                                             const TensorLayout&,
                                             const TensorLayout&,
                                             const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())
                    ->default_batch_conv_bias_fwd_algo()};
}

BatchConvBiasForward::Algorithm*
BatchConvBiasForwardImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* filter */,
        const TensorLayout& /* bias */, const TensorLayout& /* z */,
        const TensorLayout& /* dst */, size_t /* workspace_limit_in_bytes */
        ,
        bool reproducible) {
    auto algo = static_cast<HandleImpl*>(handle())
            ->default_batch_conv_bias_fwd_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

// vim: syntax=cpp.doxygen
