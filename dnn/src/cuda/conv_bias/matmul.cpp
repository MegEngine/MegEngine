/**
 * \file dnn/src/cuda/conv_bias/matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/conv_bias/matmul/im2col.cuh"
#include "src/cuda/utils.h"
#include "src/common/algo_base.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

namespace {
std::pair<TensorLayoutArray, MatrixMulForward::Param> sub_opr_config(
        const ConvBiasForwardImpl::CanonizedFilterMeta& fm,
        const TensorLayout& src_layout, const TensorLayout& filter_layout,
        const TensorLayout& dst_layout, const ConvBiasForwardImpl* opr) {
    size_t N = src_layout.shape[0], IC = fm.icpg, OC = fm.ocpg,
           OH = dst_layout.shape[2], OW = dst_layout.shape[3],
           FH = fm.spatial[0], FW = fm.spatial[1];

    megdnn_assert(src_layout.dtype.category() == DTypeCategory::FLOAT);
    TensorLayout Al({OC, IC * FH * FW}, filter_layout.dtype),
            Bl({IC * FH * FW, OH * OW * N}, filter_layout.dtype),
            Cl({OC, OH * OW * N}, filter_layout.dtype);
    MatrixMulForward::Param param;
    if (opr->param().compute_mode == param::Convolution::ComputeMode::FLOAT32) {
        param.compute_mode = param::MatrixMul::ComputeMode::FLOAT32;
    }

    return {{Al, Bl, Cl}, param};
}

std::pair<TensorLayoutArray, std::unique_ptr<MatrixMulForward>> prepare_sub_opr(
        const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
    auto matmul_opr = args.handle->create_operator<MatrixMulForward>();
    set_execution_policy<ConvBiasForward, MatrixMulForward*>(args.opr,
                                                              matmul_opr.get());
    auto&& config =
            sub_opr_config(args.filter_meta, *args.src_layout,
                           *args.filter_layout, *args.dst_layout, args.opr);
    matmul_opr->param() = config.second;

    return {config.first, std::move(matmul_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvBiasForwardImpl::AlgoMatmul::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const ConvBiasForwardImpl* conv_bias_opr =
            static_cast<const ConvBiasForwardImpl*>(opr);
    CanonizedFilterMeta fm =
            conv_bias_opr->check_layout_fwd(layouts[0], layouts[1], layouts[4]);
    auto&& config = sub_opr_config(fm, layouts[0], layouts[1], layouts[4],
                                   conv_bias_opr);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::MATRIX_MUL_FORWARD, param_str, config.first}};
}

bool ConvBiasForwardImpl::AlgoMatmul::is_available(const SizeArgs& args) const {
    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    if (args.z_layout->ndim > 0)
        return false;

    auto&& fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
           args.src_layout->dtype.category() == DTypeCategory::FLOAT &&
           fm.group == 1 && fm.spatial_ndim == 2;
}

WorkspaceBundle ConvBiasForwardImpl::AlgoMatmul::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto dst_layout = *args.dst_layout;
    SmallVector<size_t> sizes;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
        sizes.push_back(dst_layout.span().dist_byte());
    }

    SizeArgs conv_args = args;
    conv_args.dst_layout = &dst_layout;
    SmallVector<size_t> matmul_sizes = matmul_get_workspace_bundle(conv_args);

    auto config = prepare_sub_opr(args);
    size_t mm_ws = config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]);
    matmul_sizes.push_back(mm_ws);

    sizes.insert(sizes.begin(), matmul_sizes.begin(), matmul_sizes.end());
    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::AlgoMatmul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoMatmul::exec(const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto conv_dst_tensor = *args.dst_tensor;
    if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        conv_dst_tensor.raw_ptr = bundle.get(bundle.nr_workspace() - 1);
        conv_dst_tensor.layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            conv_dst_tensor.layout.dtype);
    }

    ExecArgs conv_args = args;
    conv_args.dst_tensor = &conv_dst_tensor;
    {
        switch (conv_args.src_layout->dtype.enumv()) {
#define cb(dt)                                        \
    case DTypeTrait<dt>::enumv: {                     \
        using ctype = typename DTypeTrait<dt>::ctype; \
        exec_internal<ctype>(conv_args, bundle);      \
        break;                                        \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

template <typename T>
void ConvBiasForwardImpl::AlgoMatmul::exec_internal(
        const ExecArgs& args, const WorkspaceBundle& bundle) {
    auto&& fm = args.filter_meta;
    size_t N = args.src_layout->shape[0], IC = fm.icpg,
           IH = args.src_layout->shape[2], IW = args.src_layout->shape[3],
           OC = fm.ocpg, OH = args.dst_tensor->layout.shape[2],
           OW = args.dst_tensor->layout.shape[3], FH = fm.spatial[0],
           FW = fm.spatial[1], PH = fm.padding[0], PW = fm.padding[1],
           SH = fm.stride[0], SW = fm.stride[1], DH = fm.dilation[0],
           DW = fm.dilation[1];
    auto stream = cuda_stream(args.handle);
    T* dst_t = static_cast<T*>(bundle.get(0));
    T* col = static_cast<T*>(bundle.get(1));
    conv_bias::im2col<T>(args.src_tensor->ptr<T>(), col, N,
                         args.src_layout->stride[0], IC, IH, IW, FH, FW, OH, OW,
                         PH, PW, SH, SW, DH, DW, stream);

    auto config = prepare_sub_opr(args);

    TensorND A(args.filter_tensor->ptr<T>(), config.first[0]),
            B(col, config.first[1]), C(dst_t, config.first[2]);
    size_t matmul_ws_idx = 2;
    if (fm.should_flip) {
        conv_bias::flip_filter(args, bundle.get_workspace(2), A.raw_ptr);
        matmul_ws_idx = 3;
    }

    config.second->exec(A, B, C, bundle.get_workspace(matmul_ws_idx));

    TensorLayout C2l({OC * OH * OW, N}, typename DTypeTrait<T>::dtype()),
            C3l = C2l;
    C3l.stride[0] = 1;
    C3l.stride[1] = args.dst_tensor->layout.stride[0];
    TensorND C2(dst_t, C2l);
    TensorND C3(args.dst_tensor->ptr<T>(), C3l);
    args.handle->relayout_opr()->exec(C2, C3);
}

// vim: syntax=cpp.doxygen
