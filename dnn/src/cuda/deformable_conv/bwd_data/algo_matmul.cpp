/**
 * \file dnn/src/cuda/deformable_conv/bwd_data/algo_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/utils.h"

#include "src/common/algo_base.h"
#include "src/cuda/deformable_conv/bwd_data/algo.h"
#include "src/cuda/deformable_conv/kimpl/deformable_conv.cuh"
#include "src/cuda/deformable_conv/opr_impl.h"

using namespace megdnn;
using namespace cuda;

using Algo = DeformableConvBackwardDataImpl::AlgoMatmul;
using OprParam = DeformableConvBase::Param;

namespace {
deformable_conv::Param create_param(
        const Algo::SizeArgs& args, const OprParam& opr_param, cublasHandle_t handle,
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
        const DeformableConvForwardImpl::CanonizedFilterMeta& fm,
        const TensorLayout& im, const TensorLayout& out_grad) {
    auto&& dt = im.dtype;
    size_t batch_sz = im[0], OH = out_grad[2], OW = out_grad[3], FH = fm.spatial[0],
           FW = fm.spatial[1];

    size_t M = fm.icpg * FH * FW, K = fm.ocpg, N = batch_sz * OH * OW, batch = fm.group;
    TensorLayout al = {{batch, K, M}, dt};
    TensorLayout bl = {{batch, K, N}, dt};
    TensorLayout cl = {{batch, M, N}, dt};

    BatchedMatrixMulForward::Param param;
    param.compute_mode = param::MatrixMul::ComputeMode::DEFAULT;
    param.transposeA = true;

    return {{al, bl, cl}, param};
}

std::pair<TensorLayoutArray, std::unique_ptr<BatchedMatrixMulForward>> prepare_sub_opr(
        const DeformableConvBackwardDataImpl::AlgoBase::SizeArgs& args) {
    auto bmatmul_opr = args.handle->create_operator<BatchedMatrixMulForward>();
    set_execution_policy<DeformableConvBackwardData, BatchedMatrixMulForward*>(
            args.opr, bmatmul_opr.get());

    auto&& config =
            sub_opr_config(args.filter_meta, args.im_layout, args.out_grad_layout);
    bmatmul_opr->param() = config.second;

    return {config.first, std::move(bmatmul_opr)};
}

};  // anonymous namespace

std::vector<Algorithm::SearchItem> Algo::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const DeformableConvBackwardDataImpl* deformable_conv =
            static_cast<const DeformableConvBackwardDataImpl*>(opr);
    CanonizedFilterMeta fm = deformable_conv->make_canonized_filter_meta(
            layouts[0].ndim, layouts[1], layouts[2]);
    auto&& config = sub_opr_config(fm, layouts[0], layouts[4]);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::BATCHED_MATRIX_MUL_FORWARD, param_str, config.first}};
}

bool Algo::is_available(const SizeArgs&) const {
    return true;
}

WorkspaceBundle Algo::get_bundle(const SizeArgs& args) {
    auto&& fm = args.filter_meta;
    size_t batch_sz = args.im_layout[0], IC = fm.group * fm.icpg,
           OC = args.out_grad_layout[1], OH = args.out_grad_layout[2],
           OW = args.out_grad_layout[3], FH = fm.spatial[0], FW = fm.spatial[1];

    auto config = prepare_sub_opr(args);

    size_t bmm_ws = config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]);
    size_t result_ws = batch_sz * IC * FH * FW * OH * OW * sizeof(float);
    size_t relayout_ws1 = batch_sz * OC * OH * OW * sizeof(float);
    size_t relayout_ws2 = batch_sz * IC * FH * FW * OH * OW * sizeof(float);

    return {nullptr, {bmm_ws, result_ws, relayout_ws1, relayout_ws2}};
}

size_t Algo::get_workspace_in_bytes(const SizeArgs& args) const {
    return get_bundle(args).total_size_in_bytes();
}

void Algo::exec(const ExecArgs& args) const {
    auto&& opr = args.opr;
    auto&& handle = concrete_handle(opr->handle());
    auto&& param = opr->param();
    auto p = create_param(args, param, handle->cublas_handle(), handle->stream());
    auto bundle = get_bundle(args);
    bundle.set(args.workspace.raw_ptr);

    float* dev_im = args.im_tensor.ptr<float>();
    float* dev_filter = args.filter_tensor.ptr<float>();
    float* dev_offset = args.offset_tensor.ptr<float>();
    float* dev_mask = args.mask_tensor.ptr<float>();
    float* dev_out_grad = args.out_grad_tensor.ptr<float>();

    float* dev_im_grad = args.im_grad_tensor.ptr<float>();
    float* dev_offset_grad = args.offset_grad_tensor.ptr<float>();
    float* dev_mask_grad = args.mask_grad_tensor.ptr<float>();

    void* bmm_ws = bundle.get(0);
    float* result_ws = static_cast<float*>(bundle.get(1));
    float* relayout_ws1 = static_cast<float*>(bundle.get(2));

    // clear out grad
    {
        size_t im_sz = p.batch_sz * p.IC * p.IH * p.IW * sizeof(float);
        size_t offset_sz = p.batch_sz * 2 * p.deformable_group * p.FH * p.FW * p.OH *
                           p.OW * sizeof(float);
        size_t mask_sz = p.batch_sz * p.deformable_group * p.FH * p.FW * p.OH * p.OW *
                         sizeof(float);

        cudaMemsetAsync(dev_im_grad, 0, im_sz, p.stream);
        cudaMemsetAsync(dev_offset_grad, 0, offset_sz, p.stream);
        cudaMemsetAsync(dev_mask_grad, 0, mask_sz, p.stream);
    }

    // relayout out_grad to [oc, N, OH, OW]
    {
        auto&& dt = args.im_layout.dtype;
        size_t dim0 = p.batch_sz, dim1 = p.OC, dim2 = p.OH * p.OW;
        TensorLayout C2l({dim0, dim1, dim2}, dt), C3l = C2l;
        C3l.stride[0] = dim2;
        C3l.stride[1] = dim0 * dim2;
        C3l.stride[2] = 1;
        TensorND C2(dev_out_grad, C2l);
        TensorND C3(relayout_ws1, C3l);

        args.handle->relayout_opr()->exec(C2, C3);
    }
    // matmul [g, icpg, FH, FW, ocpg] * [g, ocpg, N, OH, OW] =>
    //        => [g, icpg, FH, FW, N, OH, OW]
    {
        auto config = prepare_sub_opr(args);

        TensorND A(static_cast<void*>(dev_filter), config.first[0]),
                B(static_cast<void*>(relayout_ws1), config.first[1]),
                C(static_cast<void*>(result_ws), config.first[2]);

        size_t bmm_ws_size = bundle.get_size(0);
        config.second->exec(
                A, B, C, Workspace(static_cast<megdnn::dt_byte*>(bmm_ws), bmm_ws_size));
    }
    col2im(result_ws, dev_offset, dev_mask, dev_im_grad, p);
    // col [IC, FH * FW, N, OH * OW]
    col2im_coord(
            dev_im, result_ws, dev_offset, dev_mask, dev_offset_grad, dev_mask_grad, p);
}

// vim: syntax=cpp.doxygen
