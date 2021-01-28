/**
 * \file dnn/src/cuda/convolution/backward_data/matmul.cpp
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
#include "src/cuda/matrix_mul/opr_impl.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

namespace {
std::pair<TensorLayoutArray, MatrixMulForward::Param> sub_opr_config(
        const ConvolutionBackwardDataImpl::CanonizedFilterMeta& fm,
        const TensorLayout& filter_layout, const TensorLayout& diff_layout,
        const TensorLayout& grad_layout,
        const ConvolutionBackwardDataImpl* opr) {
    size_t N = grad_layout.shape[0], IC = fm.icpg,
           OC = fm.ocpg, OH = diff_layout.shape[2],
           OW = diff_layout.shape[3], FH = fm.spatial[0],
           FW = fm.spatial[1];

    megdnn_assert(filter_layout.dtype.enumv() == diff_layout.dtype.enumv());
    TensorLayout Al({OC, IC * FH * FW}, filter_layout.dtype),
            Bl({IC * FH * FW, OH * OW * N}, filter_layout.dtype),
            Cl({OC, OH * OW * N}, filter_layout.dtype);
    MatrixMulForward::Param param;
    if (opr->param().compute_mode ==
        param::Convolution::ComputeMode::FLOAT32) {
        param.compute_mode = param::MatrixMul::ComputeMode::FLOAT32;
    }

    param.transposeA = true;
    return {{Al, Cl, Bl}, param};
}

std::pair<TensorLayoutArray, std::unique_ptr<MatrixMulForward>> prepare_sub_opr(
        const ConvolutionBackwardDataImpl::AlgoBase::SizeArgs& args) {
    auto matmul_opr = args.handle->create_operator<MatrixMulForward>();
    set_execution_policy<ConvolutionBackwardData, MatrixMulForward*>(
            args.opr, matmul_opr.get());
    auto&& config =
            sub_opr_config(args.filter_meta, *args.filter_layout,
                           *args.diff_layout, *args.grad_layout, args.opr);
    matmul_opr->param() = config.second;

    return {config.first, std::move(matmul_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvolutionBackwardDataImpl::AlgoMatmul::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const ConvolutionBackwardDataImpl* conv_backward_data_opr =
            static_cast<const ConvolutionBackwardDataImpl*>(opr);
    CanonizedFilterMeta fm = conv_backward_data_opr->check_layout_fwd(
            layouts[2], layouts[0], layouts[1]);
    auto&& config = sub_opr_config(fm, layouts[0], layouts[1], layouts[2],
                                   conv_backward_data_opr);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::MATRIX_MUL_FORWARD, param_str, config.first}};
}

bool ConvolutionBackwardDataImpl::AlgoMatmul::is_available(
        const SizeArgs& args) const {
    if (args.diff_layout->dtype == args.filter_layout->dtype &&
        args.diff_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    auto&& fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
           args.diff_layout->dtype.category() == DTypeCategory::FLOAT &&
           fm.group == 1 && fm.spatial_ndim == 2;
}

size_t ConvolutionBackwardDataImpl::AlgoMatmul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);

    auto&& sizes = matmul_get_workspace_bundle(args.as_fwd_args());
    sizes.push_back(config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]));
    return WorkspaceBundle(nullptr, sizes).total_size_in_bytes();
}

void ConvolutionBackwardDataImpl::AlgoMatmul::exec(const ExecArgs& args) const {
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
void ConvolutionBackwardDataImpl::AlgoMatmul::exec_internal(
        const ExecArgs& args) {
    auto&& fm = args.filter_meta;
    size_t N = args.grad_layout->shape[0], IC = fm.icpg,
           IH = args.grad_layout->shape[2], IW = args.grad_layout->shape[3],
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
        // take gemm grad
        TensorLayout Al({OC, IC * FH * FW}, typename DTypeTrait<T>::dtype()),
                Bl({IC * FH * FW, OH * OW * N},
                   typename DTypeTrait<T>::dtype()),
                Cl({OC, OH * OW * N}, typename DTypeTrait<T>::dtype());
        TensorND A(args.filter_tensor->ptr<T>(), Al), B(col, Bl), C(diff_t, Cl);
        if (fm.should_flip) {
            convolution::flip_filter(args.as_fwd_args(),
                                     wbundle.get_workspace(2), A.raw_ptr);
            config.second->exec(A, C, B, wbundle.get_workspace(3));
        } else {
            config.second->exec(A, C, B, wbundle.get_workspace(2));
        }
    }
    {
        // col2im
        convolution::col2im<T>(col, args.grad_tensor->ptr<T>(), N,
                               args.grad_layout->stride[0], IC, IH, IW, FH, FW,
                               OH, OW, PH, PW, SH, SW, DH, DW, stream);
    }
}

// vim: syntax=cpp.doxygen
