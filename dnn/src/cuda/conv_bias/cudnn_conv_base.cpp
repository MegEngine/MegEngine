/**
 * \file dnn/src/cuda/conv_bias/cudnn_conv_base.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

WorkspaceBundle ConvBiasForwardImpl::AlgoCUDNNConvBase::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto dst_layout = *args.dst_layout;
    SmallVector<size_t> sizes;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(
                args.src_layout->dtype, args.filter_layout->dtype, dst_layout.dtype);
        sizes.push_back(dst_layout.span().dist_byte());
    }

    if (args.z_layout->ndim > 0 &&
        args.z_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        auto z_layout = *args.z_layout;
        z_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(
                args.src_layout->dtype, args.filter_layout->dtype, z_layout.dtype);
        sizes.push_back(z_layout.span().dist_byte());
    }

    SizeArgs conv_args = args;
    conv_args.dst_layout = &dst_layout;

    size_t conv_workspace_size = cudnn_get_workspace_in_bytes(conv_args);

    sizes.insert(sizes.begin(), conv_workspace_size);
    return {ptr, std::move(sizes)};
}

void ConvBiasForwardImpl::AlgoCUDNNConvBase::exec(const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    TensorND conv_dst_tensor = *args.dst_tensor;
    if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        conv_dst_tensor = TensorND{bundle.get(1), args.dst_tensor->layout};
        conv_dst_tensor.layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(
                args.src_layout->dtype, args.filter_layout->dtype,
                conv_dst_tensor.layout.dtype);
    }

    ExecArgs conv_args = args;
    conv_args.dst_tensor = &conv_dst_tensor;
    conv_args.dst_layout = &conv_dst_tensor.layout;

    cudnn_execute(conv_args, bundle.get_workspace(0));

    if (args.z_layout->ndim > 0) {
        auto z_tensor = *args.z_tensor;
        if (args.z_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
            z_tensor = TensorND{bundle.get(2), args.z_tensor->layout};
            z_tensor.layout.dtype = DType();
            args.opr->check_or_deduce_dtype_fwd(
                    args.src_layout->dtype, args.filter_layout->dtype,
                    z_tensor.layout.dtype);
            auto typecvt = args.handle->create_operator<TypeCvt>();
            typecvt->exec(*args.z_tensor, z_tensor);
        }
        auto add = args.handle->create_operator<ElemwiseForward>();
        add->param().mode = Elemwise::Param::Mode::ADD;
        add->exec({conv_dst_tensor, z_tensor}, conv_dst_tensor);
    }

    handle_bias_and_nonlinear(
            args.handle, args.nonlinear_mode, &conv_dst_tensor, args.dst_tensor,
            args.bias_tensor);
}

// vim: syntax=cpp.doxygen
