/**
 * \file dnn/src/cuda/deformable_conv/bwd_flt/algo_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/utils.h"

#include "src/cuda/deformable_conv/bwd_flt/algo.h"
#include "src/cuda/deformable_conv/kimpl/deformable_conv.cuh"
#include "src/cuda/deformable_conv/opr_impl.h"
#include "src/common/algo_base.h"

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

std::pair<TensorLayoutArray, BatchedMatrixMulForward::Param> sub_opr_config(
        const DeformableConvBackwardFilterImpl::CanonizedFilterMeta& fm,
        const TensorLayout& im, const TensorLayout& out_grad) {
    auto&& dt = im.dtype;
    size_t batch_sz = im[0], OH = out_grad[2], OW = out_grad[3],
           FH = fm.spatial[0], FW = fm.spatial[1];

    size_t M = fm.ocpg, K = OH * OW * batch_sz, N = fm.icpg * FH * FW,
           batch = fm.group;
    TensorLayout al = {{batch, M, K}, dt};
    TensorLayout bl = {{batch, N, K}, dt};
    TensorLayout cl = {{batch, M, N}, dt};

    BatchedMatrixMulForward::Param param;
    param.compute_mode = param::MatrixMul::ComputeMode::DEFAULT;
    param.transposeB = true;

    return {{al, bl, cl}, param};
}

std::pair<TensorLayoutArray, std::unique_ptr<BatchedMatrixMulForward>>
prepare_sub_opr(
        const DeformableConvBackwardFilterImpl::AlgoBase::SizeArgs& args) {
    auto bmatmul_opr = args.handle->create_operator<BatchedMatrixMulForward>();
    set_execution_policy<DeformableConvBackwardFilter,
                         BatchedMatrixMulForward*>(args.opr, bmatmul_opr.get());

    auto&& config = sub_opr_config(args.filter_grad_meta, args.im_layout,
                                   args.out_grad_layout);
    bmatmul_opr->param() = config.second;

    return {config.first, std::move(bmatmul_opr)};
}

};  // anonymous namespace

std::vector<Algorithm::SearchItem> Algo::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const DeformableConvBackwardFilterImpl* deformable_conv =
            static_cast<const DeformableConvBackwardFilterImpl*>(opr);
    CanonizedFilterMeta fm = deformable_conv->make_canonized_filter_meta(
            layouts[0].ndim, layouts[4], layouts[1]);
    auto&& config = sub_opr_config(fm, layouts[0], layouts[3]);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::BATCHED_MATRIX_MUL_FORWARD, param_str,
             config.first}};
}

bool Algo::is_available(const SizeArgs&) const {
    return true;
}

WorkspaceBundle Algo::get_bundle(const SizeArgs& args) {
    auto&& fm = args.filter_grad_meta;
    auto OH = args.out_grad_layout[2], OW = args.out_grad_layout[3];
    auto FH = fm.spatial[0], FW = fm.spatial[1];
    size_t IC = fm.group * fm.icpg, OC = args.out_grad_layout[1];
    auto batch_sz = args.im_layout[0];

    auto config = prepare_sub_opr(args);

    size_t col_ws = batch_sz * IC * FH * FW * OH * OW * sizeof(float);
    size_t out_grad_ws = batch_sz * OC * OH * OW * sizeof(float);
    size_t bmm_ws = config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]);

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
    auto config = prepare_sub_opr(args);

    TensorND A(static_cast<void*>(out_grad_ws), config.first[0]),
            B(static_cast<void*>(col_ws), config.first[1]),
            C(static_cast<void*>(dev_filter_grad), config.first[2]);

    size_t bmm_ws_size = bundle.get_size(2);
    config.second->exec(
            A, B, C,
            Workspace(static_cast<megdnn::dt_byte*>(bmm_ws), bmm_ws_size));
}
// vim: syntax=cpp.doxygen
