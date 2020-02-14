/**
 * \file dnn/src/cuda/local_share/forward/batched_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./algo.h"
#include "src/cuda/local_share/im2col.cuh"
#include "src/cuda/local_share/opr_impl.h"

using namespace megdnn;
using namespace cuda;

bool LocalShareForwardImpl::AlgoBatchedMatMul::is_available(
        const SizeArgs& args) const {
    bool available = true;
    auto&& param = args.opr->param();
    using Param = LocalShare::Param;
    using Format = Param::Format;
    // NCHW format
    available &= param.format == Format::NCHW;
    // only support float
    auto src_dtype = args.src_layout.dtype,
         filter_dtype = args.filter_layout.dtype,
         dst_dtype = args.dst_layout.dtype;
    available &= (src_dtype == filter_dtype) && (src_dtype == dst_dtype) &&
                 (src_dtype == dtype::Float32());
    // do not support dilate conv
    size_t dh = param.dilate_h, dw = param.dilate_w;
    available &= (dh == 1 && dw == 1);
    return available;
}

WorkspaceBundle LocalShareForwardImpl::AlgoBatchedMatMul::get_workspace_bundle(
        dt_byte* raw_ptr, const SizeArgs& args) const {
    auto&& param = args.opr->param();
    unpack_local_share_params(args.src_layout, args.filter_layout,
                              args.dst_layout, param);
    using Param = LocalShare::Param;
    using Sparse = Param::Sparse;
    size_t groups = 1;
    if (param.sparse == Sparse::GROUP) {
        groups = args.filter_layout.shape[0];
    }
    size_t icpg = ci / groups, ocpg = co / groups;
    size_t ws_im2col =
            n * ci * ho * wo * fh * fw * args.src_layout.dtype.size();
    size_t ws_posttranspose = n * co * ho * wo * args.dst_layout.dtype.size();
    auto&& matmul_opr = args.opr->handle()->create_operator<BatchedMatrixMul>();
    TensorLayout A{
            {groups * sgh * sgw, ho / sgh * wo / sgw * n, icpg * fh * fw},
            dtype::Float32()};
    TensorLayout B{{groups * sgh * sgw, icpg * fh * fw, ocpg},
                   dtype::Float32()};
    TensorLayout C{{groups * sgh * sgw, ho / sgh * wo / sgw * n, ocpg},
                   dtype::Float32()};
    size_t ws_matmul = matmul_opr->get_workspace_in_bytes(A, B, C);
    WorkspaceBundle ws{raw_ptr, {ws_im2col, ws_matmul, ws_posttranspose}};
    return ws;
}

size_t LocalShareForwardImpl::AlgoBatchedMatMul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void LocalShareForwardImpl::AlgoBatchedMatMul::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    unpack_local_share_params(args.src_layout, args.filter_layout,
                              args.dst_layout, param);
    using Param = LocalShare::Param;
    using Sparse = Param::Sparse;
    size_t groups = 1;
    if (param.sparse == Sparse::GROUP) {
        groups = args.filter_layout.shape[0];
    }
    size_t icpg = ci / groups, ocpg = co / groups;
    local_share::Param kern_param;
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ph = ph,
    kern_param.pw = pw, kern_param.grp_ho = ho / sgh,
    kern_param.grp_wo = wo / sgw, kern_param.sgh = sgh, kern_param.sgw = sgw;

    auto ws = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto ws_im2col = ws.get(0);
    auto ws_matmul = ws.get(1);
    auto ws_posttranspose = ws.get(2);
    auto&& stream = cuda_stream(args.opr->handle());
    local_share::_do_local_share_im2col(
            args.src_tensor->ptr<dt_float32>(),
            reinterpret_cast<dt_float32*>(ws_im2col), fh, fw, sh, sw, groups,
            kern_param, stream);

    auto&& matmul_opr = args.opr->handle()->create_operator<BatchedMatrixMul>();
    TensorLayout A{
            {groups * sgh * sgw, ho / sgh * wo / sgw * n, icpg * fh * fw},
            dtype::Float32()};
    TensorLayout B{{groups * sgh * sgw, icpg * fh * fw, ocpg},
                   dtype::Float32()};
    TensorLayout C{{groups * sgh * sgw, ho / sgh * wo / sgw * n, ocpg},
                   dtype::Float32()};
    TensorND ts_A{ws_im2col, A};
    TensorND ts_B{args.filter_tensor->raw_ptr, B};
    TensorND ts_C{ws_posttranspose, C};
    Workspace ws_wrapper;
    ws_wrapper.raw_ptr = reinterpret_cast<dt_byte*>(ws_matmul);
    ws_wrapper.size = ws.get_size(1);
    matmul_opr->exec(ts_A, ts_B, ts_C, ws_wrapper);

    {
        TensorLayout C1{{n, groups, ocpg, sgh, ho / sgh, sgw, wo / sgw},
                        dtype::Float32()};
        C1.stride[0] = ho / sgh * wo / sgw * ocpg;
        C1.stride[1] = n * ho * wo * ocpg;
        C1.stride[2] = 1;
        C1.stride[3] = n * ho / sgh * wo * ocpg;
        C1.stride[4] = wo / sgw * ocpg;
        C1.stride[5] = n * ho / sgh * wo / sgw * ocpg;
        C1.stride[6] = ocpg;
        TensorLayout C2 = args.dst_layout;
        TensorND ts_C1{ws_posttranspose, C1};
        TensorND ts_C2{args.dst_tensor->raw_ptr, C2};
        auto&& relayout_opr = args.opr->handle()->create_operator<Relayout>();
        relayout_opr->exec(ts_C1, ts_C2);
    }
}

// vim: syntax=cpp.doxygen
