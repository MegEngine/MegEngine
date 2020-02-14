/**
 * \file dnn/src/cuda/local_share/backward_data/implicit_gemm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./algo.h"
#include "./local_share_bwd_data.cuh"
#include "src/cuda/local_share/opr_impl.h"

#include <cstring>
#include "src/common/utils.h"

using namespace megdnn;
using namespace cuda;

bool LocalShareBackwardDataImpl::AlgoImplicitGemm::is_available(
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
    unpack_local_share_params(args.grad_layout, args.filter_layout,
                              args.diff_layout, param);
    available &= (ho % sgh == 0 && wo % sgw == 0);
    // not support dilated convolution
    available &= (dh == 1 && dw == 1);
    available &= (co % 4 == 0);
    auto filter_dtype = args.filter_layout.dtype,
         diff_dtype = args.diff_layout.dtype,
         grad_dtype = args.grad_layout.dtype;
    // only support float32
    available &= (filter_dtype == diff_dtype && filter_dtype == grad_dtype &&
                  filter_dtype == dtype::Float32());
    // only support sm_60 or later
    available &= is_compute_capability_required(6, 0);

    return available;
}

size_t
LocalShareBackwardDataImpl::AlgoImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    unpack_local_share_params(args.grad_layout, args.filter_layout,
                              args.diff_layout, param);
    size_t ws_size_grad = n * ci * hi * wi * args.grad_layout.dtype.size();
    size_t ws_size_diff = n * co * ho * wo * args.diff_layout.dtype.size();
    return ws_size_grad + ws_size_diff;
}

void LocalShareBackwardDataImpl::AlgoImplicitGemm::exec(
        const ExecArgs& args) const {
    local_share::Param kern_param;
    auto&& param = args.opr->param();
    unpack_local_share_params(args.grad_layout, args.filter_layout,
                              args.diff_layout, param);
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ph = ph,
    kern_param.pw = pw, kern_param.grp_ho = ho / sgh,
    kern_param.grp_wo = wo / sgw, kern_param.sgh = sgh, kern_param.sgw = sgw;
    auto&& handle = concrete_handle(args.opr->handle());
    auto&& cublas_hdl = cublas_handle(args.opr->handle());
    auto&& stream = cuda_stream(args.opr->handle());

    auto one = handle->one_device();
    auto zero = handle->zero_device();

    local_share_bwd_data::_do_local_share_bwd_data_implicit_gemm(
            args.filter_tensor->ptr<dt_float32>(),
            args.diff_tensor->ptr<dt_float32>(),
            args.grad_tensor->ptr<dt_float32>(),
            reinterpret_cast<float*>(args.workspace.raw_ptr), fh, fw, sh, sw,
            kern_param, cublas_hdl, stream, one, zero);
}

// vim: syntax=cpp.doxygen
