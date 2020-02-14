/**
 * \file dnn/src/cuda/local_share/forward/batch_size_aware_chwn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./algo.h"
#include "./local_share_forward.cuh"
#include "src/cuda/local_share/opr_impl.h"

#include <cstring>
#include "src/common/utils.h"

using namespace megdnn;
using namespace cuda;

bool LocalShareForwardImpl::AlgoCHWNBatchSizeAware::is_available(
        const SizeArgs& args) const {
    using Param = LocalShare::Param;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    auto&& param = args.opr->param();
    auto format = param.format;
    auto sparse = param.sparse;
    auto mode = param.mode;
    bool available = true;
    // format must be nchw
    available &= (format == Format::NCHW);
    // only support dense conv
    available &= (sparse == Sparse::DENSE);
    // mode must be cross correlation
    available &= (mode == Mode::CROSS_CORRELATION);
    unpack_local_share_params(args.src_layout, args.filter_layout,
                              args.dst_layout, param);
    available &= (ho % sgh == 0 && wo % sgw == 0);
    // not support dilated convolution
    available &= (dh == 1 && dw == 1);
    available &= (n % 32 == 0);
    // kernel size should be 3, 5, 7
    available &= (fh == 1 && fw == 1) || (fh == 3 && fw == 3) ||
                 (fh == 5 && fw == 5) || (fh == 7 || fw == 7);
    // stride should be 1 or 2
    available &= (sh == sw && (sh == 1 || sh == 2));
    available &= (ci % 4 == 0) || (fh == 3 && ci % 2 == 0);
    auto src_dtype = args.src_layout.dtype,
         filter_dtype = args.filter_layout.dtype,
         dst_dtype = args.dst_layout.dtype;
    // only support float32
    available &= (src_dtype == filter_dtype && src_dtype == dst_dtype &&
                  src_dtype == dtype::Float32());
    // only support sm_60 or later
    available &= is_compute_capability_required(6, 0);

    return available;
}

WorkspaceBundle
LocalShareForwardImpl::AlgoCHWNBatchSizeAware::get_workspace_bundle(
        dt_byte* raw_ptr, const SizeArgs& args) const {
    auto&& param = args.opr->param();
    unpack_local_share_params(args.src_layout, args.filter_layout,
                              args.dst_layout, param);
    size_t ws_size_src = n * ci * hi * wi * args.src_layout.dtype.size();
    size_t ws_size_dst = n * co * ho * wo * args.dst_layout.dtype.size();
    WorkspaceBundle ws{raw_ptr, {ws_size_src, ws_size_dst}};
    return ws;
}

size_t LocalShareForwardImpl::AlgoCHWNBatchSizeAware::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void LocalShareForwardImpl::AlgoCHWNBatchSizeAware::exec(
        const ExecArgs& args) const {
    local_share::Param kern_param;
    auto&& param = args.opr->param();
    unpack_local_share_params(args.src_layout, args.filter_layout,
                              args.dst_layout, param);
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ph = ph,
    kern_param.pw = pw, kern_param.grp_ho = ho / sgh,
    kern_param.grp_wo = wo / sgw, kern_param.sgh = sgh, kern_param.sgw = sgw;
    auto&& handle = concrete_handle(args.opr->handle());
    auto&& cublas_hdl = cublas_handle(args.opr->handle());
    auto&& stream = cuda_stream(args.opr->handle());

    auto one = handle->one_device();
    auto zero = handle->zero_device();

    local_share::_do_local_share_convolution_large_batch_size(
            args.src_tensor->ptr<dt_float32>(),
            args.filter_tensor->ptr<dt_float32>(),
            args.dst_tensor->ptr<dt_float32>(),
            reinterpret_cast<float*>(args.workspace.raw_ptr), fh, fw, sh, sw,
            kern_param, cublas_hdl, stream, one, zero);
}

// vim: syntax=cpp.doxygen
