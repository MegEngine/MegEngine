/**
 * \file dnn/src/cuda/deformable_conv/bwd_flt/algo_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/utils.h"

#include "src/cuda/deformable_conv/bwd_flt/algo.h"
#include "src/cuda/deformable_conv/kimpl/deformable_conv.cuh"
#include "src/cuda/deformable_conv/opr_impl.h"

using namespace megdnn;
using namespace cuda;

using Algo = DeformableConvBackwardFilterImpl::AlgoMatmul;
using OprParam = DeformableConvBase::Param;

namespace {
deformable_conv::Param create_param(const Algo::SizeArgs& args,
                                    const OprParam& opr_param,
                                    cublasHandle_t handle,
                                    cudaStream_t stream) {
    deformable_conv::Param p;
    auto&& fm = args.filter_grad_meta;

    p.handle = handle;
    p.stream = stream;
    p.group = fm.group;
    p.deformable_group = fm.deformable_group;
    p.batch_sz = args.im_layout[0];

    p.IC = args.im_layout[1];
    p.IH = args.im_layout[2];
    p.IW = args.im_layout[3];
    p.OC = args.out_grad_layout[1];
    p.OH = args.out_grad_layout[2];
    p.OW = args.out_grad_layout[3];
    p.FH = fm.spatial[0];
    p.FW = fm.spatial[1];
    p.PH = opr_param.pad_h;
    p.PW = opr_param.pad_w;
    p.SH = opr_param.stride_h;
    p.SW = opr_param.stride_w;
    p.DH = opr_param.dilate_h;
    p.DW = opr_param.dilate_w;

    p.icpg = p.IC / p.group;
    p.icpdg = p.IC / p.deformable_group;
    p.ocpg = p.OC / p.group;
    p.ocpdg = p.OC / p.deformable_group;

    return p;
}
};  // anonymous namespace

bool Algo::is_available(const SizeArgs&) const {
    return true;
}

void Algo::get_matmul_layout(const SizeArgs& args, TensorLayout& al,
                             TensorLayout& bl, TensorLayout& cl) {
    auto&& dt = args.im_layout.dtype;
    auto&& fm = args.filter_grad_meta;
    size_t batch_sz = args.im_layout[0], OH = args.out_grad_layout[2],
           OW = args.out_grad_layout[3], FH = fm.spatial[0], FW = fm.spatial[1];

    size_t M = fm.ocpg, K = OH * OW * batch_sz, N = fm.icpg * FH * FW,
           batch = fm.group;

    al = {{batch, M, K}, dt};
    bl = {{batch, N, K}, dt};
    cl = {{batch, M, N}, dt};
}

WorkspaceBundle Algo::get_bundle(const SizeArgs& args) {
    auto&& fm = args.filter_grad_meta;
    auto OH = args.out_grad_layout[2], OW = args.out_grad_layout[3];
    auto FH = fm.spatial[0], FW = fm.spatial[1];
    size_t IC = fm.group * fm.icpg, OC = args.out_grad_layout[1];
    auto batch_sz = args.im_layout[0];

    auto&& bmm_opr = args.handle->create_operator<BatchedMatrixMulForward>();
    TensorLayout al, bl, cl;

    get_matmul_layout(args, al, bl, cl);
    bmm_opr->param().compute_mode = param::MatrixMul::ComputeMode::DEFAULT;
    bmm_opr->param().transposeB = true;

    size_t col_ws = batch_sz * IC * FH * FW * OH * OW * sizeof(float);
    size_t out_grad_ws = batch_sz * OC * OH * OW * sizeof(float);
    size_t bmm_ws = bmm_opr->get_workspace_in_bytes(al, bl, cl);

    return {nullptr, {col_ws, out_grad_ws, bmm_ws}};
}

size_t Algo::get_workspace_in_bytes(const SizeArgs& args) const {
    return get_bundle(args).total_size_in_bytes();
}

void Algo::exec(const ExecArgs& args) const {
    auto&& opr = args.opr;
    auto&& param = opr->param();
    auto&& handle = concrete_handle(opr->handle());

    auto p = create_param(args, param, handle->cublas_handle(),
                          handle->stream());

    auto bundle = get_bundle(args);
    bundle.set(args.workspace.raw_ptr);

    const float* dev_im = args.im_tensor.ptr<float>();
    const float* dev_offset = args.offset_tensor.ptr<float>();
    const float* dev_mask = args.mask_tensor.ptr<float>();
    float* dev_out_grad = args.out_grad_tensor.ptr<float>();
    float* dev_filter_grad = args.filter_grad_tensor.ptr<float>();

    float* col_ws = static_cast<float*>(bundle.get(0));
    float* out_grad_ws = static_cast<float*>(bundle.get(1));
    void* bmm_ws = bundle.get(2);

    // im2col
    deformable_conv::im2col(dev_im, dev_offset, dev_mask, col_ws, p);
    // relayout
    auto&& dt = args.im_layout.dtype;
    size_t dim0 = p.batch_sz, dim1 = p.OC, dim2 = p.OH * p.OW;
    TensorLayout C2l({dim0, dim1, dim2}, dt), C3l = C2l;
    C3l.stride[0] = dim2;
    C3l.stride[1] = dim0 * dim2;
    C3l.stride[2] = 1;
    TensorND C2(dev_out_grad, C2l);
    TensorND C3(out_grad_ws, C3l);

    args.handle->relayout_opr()->exec(C2, C3);
    // matmul
    TensorLayout al, bl, cl;
    get_matmul_layout(args, al, bl, cl);

    TensorND A(static_cast<void*>(out_grad_ws), al),
            B(static_cast<void*>(col_ws), bl),
            C(static_cast<void*>(dev_filter_grad), cl);

    size_t bmm_ws_size = bundle.get_size(2);
    auto&& bmm_opr = args.handle->create_operator<BatchedMatrixMulForward>();

    bmm_opr->param().compute_mode = param::MatrixMul::ComputeMode::DEFAULT;
    bmm_opr->param().transposeB = true;

    bmm_opr->exec(
            A, B, C,
            Workspace(static_cast<megdnn::dt_byte*>(bmm_ws), bmm_ws_size));
}
// vim: syntax=cpp.doxygen
