/**
 * \file dnn/src/cuda/local_share/backward_filter/batched_matmul.cpp
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

#include <cstring>
#include "src/common/utils.h"

using namespace megdnn;
using namespace cuda;

bool LocalShareBackwardFilterImpl::AlgoBatchedMatMul::is_available(
        const SizeArgs& args) const {
    using Param = LocalShare::Param;
    using Format = Param::Format;
    using Mode = Param::Mode;
    auto&& param = args.opr->param();
    auto format = param.format;
    auto mode = param.mode;
    bool available = true;
    // format must be nchw
    available &= (format == Format::NCHW);
    // mode must be cross correlation
    available &= (mode == Mode::CROSS_CORRELATION);
    auto src_dtype = args.src_layout.dtype, diff_dtype = args.diff_layout.dtype,
         grad_dtype = args.grad_layout.dtype;
    // only support float32
    available &= (src_dtype == diff_dtype && src_dtype == grad_dtype &&
                  src_dtype == dtype::Float32());
    // do not support dilate conv
    size_t dh = param.dilate_h, dw = param.dilate_w;
    available &= (dh == 1 && dw == 1);
    return available;
}

WorkspaceBundle
LocalShareBackwardFilterImpl::AlgoBatchedMatMul::get_workspace_bundle(
        dt_byte* raw_ptr, const SizeArgs& args) const {
    auto&& param = args.opr->param();
    unpack_local_share_params(args.src_layout, args.grad_layout,
                              args.diff_layout, param);
    using Param = LocalShare::Param;
    using Sparse = Param::Sparse;
    size_t groups = 1;
    if (param.sparse == Sparse::GROUP) {
        groups = args.grad_layout.shape[0];
    }
    size_t icpg = ci / groups, ocpg = co / groups;
    size_t ws_im2col =
            n * ci * ho * wo * fh * fw * args.src_layout.dtype.size();
    size_t ws_pretranspose = n * co * ho * wo * args.diff_layout.dtype.size();
    auto&& matmul_opr = args.opr->handle()->create_operator<BatchedMatrixMul>();
    matmul_opr->param().transposeA = true;
    matmul_opr->param().transposeB = true;
    TensorLayout A{
            {groups * sgh * sgw, ho / sgh * wo / sgw * n, icpg * fh * fw},
            dtype::Float32()};
    TensorLayout B{{groups * sgh * sgw, ocpg, ho / sgh * wo / sgw * n},
                   dtype::Float32()};
    TensorLayout C{{groups * sgh * sgw, icpg * fh * fw, ocpg},
                   dtype::Float32()};
    size_t ws_matmul = matmul_opr->get_workspace_in_bytes(A, B, C);
    WorkspaceBundle ws{raw_ptr, {ws_im2col, ws_pretranspose, ws_matmul}};
    return ws;
}

size_t LocalShareBackwardFilterImpl::AlgoBatchedMatMul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void LocalShareBackwardFilterImpl::AlgoBatchedMatMul::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    unpack_local_share_params(args.src_layout, args.grad_layout,
                              args.diff_layout, param);
    using Param = LocalShare::Param;
    using Sparse = Param::Sparse;
    size_t groups = 1;
    if (param.sparse == Sparse::GROUP) {
        groups = args.grad_layout.shape[0];
    }
    size_t icpg = ci / groups, ocpg = co / groups;
    local_share::Param kern_param;
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ph = ph,
    kern_param.pw = pw, kern_param.grp_ho = ho / sgh,
    kern_param.grp_wo = wo / sgw, kern_param.sgh = sgh, kern_param.sgw = sgw;

    auto ws = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto ws_im2col = ws.get(0);
    auto ws_pretranspose = ws.get(1);
    auto ws_matmul = ws.get(2);
    auto&& stream = cuda_stream(args.opr->handle());
    local_share::_do_local_share_im2col(
            args.src_tensor->ptr<dt_float32>(),
            reinterpret_cast<dt_float32*>(ws_im2col), fh, fw, sh, sw, groups,
            kern_param, stream);

    {
        TensorLayout B1{{groups, sgh, sgw, ocpg, n, ho / sgh, wo / sgw},
                        dtype::Float32()};
        B1.stride[0] = wo * ho * ocpg;
        B1.stride[1] = wo * ho / sgh;
        B1.stride[2] = wo / sgw;
        B1.stride[3] = ho * wo;
        B1.stride[4] = co * ho * wo;
        B1.stride[5] = wo;
        B1.stride[6] = 1;
        TensorND ts_B1{args.diff_tensor->raw_ptr, B1};
        TensorLayout B2{{groups * sgh * sgw, ocpg, ho / sgh * wo / sgw * n},
                        dtype::Float32()};
        B2.init_contiguous_stride();
        TensorND ts_B2{ws_pretranspose, B2};
        auto&& relayout_opr = args.opr->handle()->create_operator<Relayout>();
        relayout_opr->exec(ts_B1, ts_B2);
    }

    auto&& matmul_opr = args.opr->handle()->create_operator<BatchedMatrixMul>();
    matmul_opr->param().transposeA = true;
    matmul_opr->param().transposeB = true;
    TensorLayout A{
            {groups * sgh * sgw, ho / sgh * wo / sgw * n, icpg * fh * fw},
            dtype::Float32()};
    TensorLayout B{{groups * sgh * sgw, ocpg, ho / sgh * wo / sgw * n},
                   dtype::Float32()};
    TensorLayout C{{groups * sgh * sgw, icpg * fh * fw, ocpg},
                   dtype::Float32()};
    TensorND ts_A{ws_im2col, A};
    TensorND ts_B{ws_pretranspose, B};
    TensorND ts_C{args.grad_tensor->raw_ptr, C};
    Workspace ws_wrapper;
    ws_wrapper.raw_ptr = reinterpret_cast<dt_byte*>(ws_matmul);
    ws_wrapper.size = ws.get_size(2);
    matmul_opr->exec(ts_A, ts_B, ts_C, ws_wrapper);
}

// vim: syntax=cpp.doxygen
