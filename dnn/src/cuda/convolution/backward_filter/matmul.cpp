/**
 * \file dnn/src/cuda/convolution/backward_filter/matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./algo.h"
#include "src/common/algo_base.h"
#include "src/cuda/convolution/helper.h"
#include "src/cuda/convolution/im2col.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

namespace {
std::pair<TensorLayoutArray, MatrixMulForward::Param> sub_opr_config(
        const ConvolutionBackwardDataImpl::CanonizedFilterMeta& fm,
        const TensorLayout& src_layout, const TensorLayout& diff_layout,
        const TensorLayout& grad_layout,
        const ConvolutionBackwardFilterImpl* opr) {
    size_t N = grad_layout.shape[0], IC = fm.icpg,
           OC = fm.ocpg, OH = diff_layout.shape[2],
           OW = diff_layout.shape[3], FH = fm.spatial[0],
           FW = fm.spatial[1];

    megdnn_assert(src_layout.dtype.enumv() == diff_layout.dtype.enumv());
    TensorLayout Al({OC, IC * FH * FW}, src_layout.dtype),
            Bl({IC * FH * FW, OH * OW * N}, src_layout.dtype),
            Cl({OC, OH * OW * N}, src_layout.dtype);
    MatrixMulForward::Param param;
    if (opr->param().compute_mode ==
        param::Convolution::ComputeMode::FLOAT32) {
        param.compute_mode = param::MatrixMul::ComputeMode::FLOAT32;
    }

    param.transposeB = true;
    return {{Cl, Bl, Al}, param};
}

std::pair<TensorLayoutArray, std::unique_ptr<MatrixMulForward>> prepare_sub_opr(
        const ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs& args) {
    auto matmul_opr = args.handle->create_operator<MatrixMulForward>();
    set_execution_policy<ConvolutionBackwardFilter, MatrixMulForward*>(
            args.opr, matmul_opr.get());

    auto&& config =
            sub_opr_config(args.grad_filter_meta, *args.src_layout,
                           *args.diff_layout, *args.grad_layout, args.opr);
    matmul_opr->param() = config.second;

    return {config.first, std::move(matmul_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvolutionBackwardFilterImpl::AlgoMatmul::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const ConvolutionBackwardFilterImpl* conv_backward_filter_opr =
            static_cast<const ConvolutionBackwardFilterImpl*>(opr);
    CanonizedFilterMeta fm = conv_backward_filter_opr->check_layout_fwd(
            layouts[0], layouts[2], layouts[1]);
    auto&& config = sub_opr_config(fm, layouts[0], layouts[1], layouts[2],
                                   conv_backward_filter_opr);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::MATRIX_MUL_FORWARD, param_str, config.first}};
}

bool ConvolutionBackwardFilterImpl::AlgoMatmul::is_available(
        const SizeArgs& args) const {
    if (args.src_layout->dtype == args.diff_layout->dtype &&
        args.diff_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    auto&& fm = args.grad_filter_meta;
    return fm.format == Param::Format::NCHW &&
           args.diff_layout->dtype.category() == DTypeCategory::FLOAT &&
           fm.group == 1 && fm.spatial_ndim == 2;
}

size_t ConvolutionBackwardFilterImpl::AlgoMatmul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);

    auto&& sizes = matmul_get_workspace_bundle(args.as_fwd_args());
    sizes.push_back(config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]));
    return WorkspaceBundle(nullptr, sizes).total_size_in_bytes();
}

void ConvolutionBackwardFilterImpl::AlgoMatmul::exec(
        const ExecArgs& args) const {
#define cb(DType)                                        \
    if (args.diff_layout->dtype == DType()) {            \
        using ctype = typename DTypeTrait<DType>::ctype; \
        exec_internal<ctype>(args);                      \
        return;                                          \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb

    megdnn_assert_internal(0);
}

template <typename T>
void ConvolutionBackwardFilterImpl::AlgoMatmul::exec_internal(
        const ExecArgs& args) {
    auto&& fm = args.grad_filter_meta;
    size_t N = args.src_layout->shape[0], IC = fm.icpg,
           IH = args.src_layout->shape[2], IW = args.src_layout->shape[3],
           OC = fm.ocpg, OH = args.diff_layout->shape[2],
           OW = args.diff_layout->shape[3], FH = fm.spatial[0],
           FW = fm.spatial[1], PH = fm.padding[0], PW = fm.padding[1],
           SH = fm.stride[0], SW = fm.stride[1], DH = fm.dilation[0],
           DW = fm.dilation[1];
    auto stream = cuda_stream(args.handle);

    auto config = prepare_sub_opr(args);

    auto&& sizes = matmul_get_workspace_bundle(args.as_fwd_args());
    sizes.push_back(config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]));
    auto wbundle = WorkspaceBundle(args.workspace.raw_ptr, sizes);

    T* diff_t = static_cast<T*>(wbundle.get(0));
    T* col = static_cast<T*>(wbundle.get(1));
    {
        // transpose diff
        TensorLayout froml({N, OC * OH * OW}, typename DTypeTrait<T>::dtype()),
                tol(froml);
        froml.stride[0] = args.diff_layout->stride[0];
        tol.stride[0] = 1;
        tol.stride[1] = N;
        TensorND from(args.diff_tensor->ptr<T>(), froml), to(diff_t, tol);
        args.handle->relayout_opr()->exec(from, to);
    }
    {
        // im2col
        convolution::im2col<T>(args.src_tensor->ptr<T>(), col, N,
                               args.src_tensor->layout.stride[0], IC, IH, IW,
                               FH, FW, OH, OW, PH, PW, SH, SW, DH, DW, stream);
    }
    {
        // take gemm grad
        TensorLayout Al({OC, IC * FH * FW}, typename DTypeTrait<T>::dtype()),
                Bl({IC * FH * FW, OH * OW * N},
                   typename DTypeTrait<T>::dtype()),
                Cl({OC, OH * OW * N}, typename DTypeTrait<T>::dtype());
        TensorND A(args.grad_tensor->ptr<T>(), Al), B(col, Bl), C(diff_t, Cl);
        if (fm.should_flip) {
            A.raw_ptr = wbundle.get(2);
            config.second->exec(C, B, A, wbundle.get_workspace(3));
            convolution::flip_filter(
                    args.as_fwd_args(),
                    {static_cast<dt_byte*>(args.grad_tensor->raw_ptr),
                     wbundle.get_size(2)},
                    A.raw_ptr);
        } else {
            config.second->exec(C, B, A, wbundle.get_workspace(2));
        }
    }
}

// vim: syntax=cpp.doxygen
