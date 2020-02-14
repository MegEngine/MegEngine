/**
 * \file dnn/src/cuda/deformable_conv/fwd/algo_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/handle.h"

#include "src/cuda/batched_matrix_mul/algo.h"
#include "src/cuda/deformable_conv/fwd/algo.h"
#include "src/cuda/deformable_conv/kimpl/deformable_conv.cuh"

using namespace megdnn;
using namespace cuda;

using Algo = DeformableConvForwardImpl::AlgoMatmul;
using OprParam = DeformableConvBase::Param;

namespace {
deformable_conv::Param create_param(const Algo::SizeArgs& args,
                                    const OprParam& opr_param,
                                    cublasHandle_t handle,
                                    cudaStream_t stream) {
    deformable_conv::Param p;
    auto&& fm = args.filter_meta;

    p.handle = handle;
    p.stream = stream;
    p.group = fm.group;
    p.deformable_group = fm.deformable_group;
    p.batch_sz = args.im_layout[0];

    p.IC = args.im_layout[1];
    p.IH = args.im_layout[2];
    p.IW = args.im_layout[3];
    p.OC = args.dst_layout[1];
    p.OH = args.dst_layout[2];
    p.OW = args.dst_layout[3];
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
    auto&& fm = args.filter_meta;
    size_t batch_sz = args.im_layout[0], OH = args.dst_layout[2],
           OW = args.dst_layout[3], FH = fm.spatial[0], FW = fm.spatial[1];

    size_t M = fm.ocpg, N = OH * OW * batch_sz, K = fm.icpg * FH * FW,
           batch = fm.group;
    al = {{batch, M, K}, dt};
    bl = {{batch, K, N}, dt};
    cl = {{batch, M, N}, dt};
}

WorkspaceBundle Algo::get_bundle(const SizeArgs& args) {
    auto&& fm = args.filter_meta;
    size_t batch_sz = args.im_layout[0], IC = fm.group * fm.icpg,
           OC = args.dst_layout[1], OH = args.dst_layout[2],
           OW = args.dst_layout[3], FH = fm.spatial[0], FW = fm.spatial[1];

    auto&& bmm_opr = args.handle->create_operator<BatchedMatrixMulForward>();
    TensorLayout al, bl, cl;

    get_matmul_layout(args, al, bl, cl);
    bmm_opr->param().compute_mode = param::MatrixMul::ComputeMode::DEFAULT;

    size_t col_ws = batch_sz * IC * FH * FW * OH * OW * sizeof(float);
    size_t bmm_ws = bmm_opr->get_workspace_in_bytes(al, bl, cl);
    size_t result_ws = batch_sz * OC * OH * OW * sizeof(float);

    return {nullptr, {col_ws, bmm_ws, result_ws}};
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

    const float* dev_im = args.im_tensor.ptr<float>();
    float* dev_filter = args.filter_tensor.ptr<float>();
    const float* dev_offset = args.offset_tensor.ptr<float>();
    const float* dev_mask = args.mask_tensor.ptr<float>();
    float* dev_out = args.dst_tensor.ptr<float>();
    void* dev_ws = args.workspace.raw_ptr;

    auto bundle = get_bundle(args);
    bundle.set(dev_ws);
    void* col_ws = bundle.get(0);
    void* bmm_ws = bundle.get(1);
    void* result_ws = bundle.get(2);
    // im2col
    deformable_conv::im2col(dev_im, dev_offset, dev_mask,
                            static_cast<float*>(col_ws), p);
    // matmul
    TensorLayout al, bl, cl;
    get_matmul_layout(args, al, bl, cl);

    TensorND A(static_cast<void*>(dev_filter), al),
            B(static_cast<void*>(col_ws), bl),
            C(static_cast<void*>(result_ws), cl);

    size_t bmm_ws_size = bundle.get_size(1);
    auto&& bmm_opr = args.handle->create_operator<BatchedMatrixMulForward>();
    bmm_opr->param().compute_mode = param::MatrixMul::ComputeMode::DEFAULT;
    bmm_opr->exec(
            A, B, C,
            Workspace(static_cast<megdnn::dt_byte*>(bmm_ws), bmm_ws_size));
    // relayout
    auto&& dt = args.im_layout.dtype;
    size_t dim0 = p.OC, dim1 = p.batch_sz, dim2 = p.OH * p.OW;
    TensorLayout C2l({dim0, dim1, dim2}, dt), C3l = C2l;
    C3l.stride[0] = dim2;
    C3l.stride[1] = dim0 * dim2;
    C3l.stride[2] = 1;
    TensorND C2(result_ws, C2l);
    TensorND C3(dev_out, C3l);

    args.handle->relayout_opr()->exec(C2, C3);
}

// vim: syntax=cpp.doxygen
