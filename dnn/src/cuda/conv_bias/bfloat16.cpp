/**
 * \file dnn/src/cuda/conv_bias/bfloat16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

ConvBiasForwardImpl::AlgoBFloat16::AlgoBFloat16(
        ConvBiasForwardImpl::AlgoBase* algorithm)
        : m_impl(algorithm) {
    megdnn_assert_internal(algorithm);
    m_name = ssprintf("BFLOAT16:%s", m_impl->name());
}

ConvBiasForwardImpl::AlgoBase::SizeArgs
ConvBiasForwardImpl::AlgoBFloat16::float_args(
        const SizeArgs& args, ConvBiasForwardImpl* opr, TensorLayout& fsrc,
        TensorLayout& ffilter, TensorLayout& fbias, TensorLayout& fz,
        TensorLayout& fdst) const {
    fsrc = *args.src_layout;
    ffilter = *args.filter_layout;
    fbias = *args.bias_layout;
    fz = *args.z_layout;
    fdst = *args.dst_layout;
    auto change_dtype = [](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
        }
    };
    change_dtype(fsrc);
    change_dtype(ffilter);
    change_dtype(fbias);
    change_dtype(fz);
    change_dtype(fdst);
    opr->param() = args.opr->param();
    opr->param().compute_mode = Param::ComputeMode::DEFAULT;
    opr->execution_policy() = {m_impl->info()};
    return SizeArgs(opr, fsrc, ffilter, fbias, fz, fdst);
}

bool ConvBiasForwardImpl::AlgoBFloat16::is_available(
        const SizeArgs& args) const {
    TensorLayout fsrc, ffilter, fbias, fz, fdst;
    auto convbias_opr = args.handle->create_operator<ConvBias>();
    SizeArgs fargs = float_args(
            args, static_cast<ConvBiasForwardImpl*>(convbias_opr.get()), fsrc,
            ffilter, fbias, fz, fdst);
    return args.src_layout->dtype == args.filter_layout->dtype &&
           args.src_layout->dtype == dtype::BFloat16() &&
           m_impl->is_available(fargs);
}

WorkspaceBundle ConvBiasForwardImpl::AlgoBFloat16::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    TensorLayout fsrc, ffilter, fbias, fz, fdst;
    auto convbias_opr = args.handle->create_operator<ConvBias>();
    SizeArgs fargs = float_args(
            args, static_cast<ConvBiasForwardImpl*>(convbias_opr.get()), fsrc,
            ffilter, fbias, fz, fdst);
    SmallVector<size_t> sizes;
    auto get_workspace = [&sizes](const TensorLayout& src,
                                  const TensorLayout& dst) {
        if (src.dtype != dst.dtype) {
            sizes.push_back(dst.span().dist_byte());
        }
    };
    get_workspace(*args.src_layout, fsrc);
    get_workspace(*args.filter_layout, ffilter);
    get_workspace(*args.bias_layout, fbias);
    get_workspace(*args.z_layout, fz);
    get_workspace(*args.dst_layout, fdst);
    sizes.push_back(m_impl->get_workspace_in_bytes(fargs));
    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::AlgoBFloat16::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoBFloat16::exec(const ExecArgs& args) const {
    TensorND fsrc_tensor = *args.src_tensor;
    TensorND ffilter_tensor = *args.filter_tensor;
    TensorND fbias_tensor = *args.bias_tensor;
    TensorND fz_tensor = *args.z_tensor;
    TensorND fdst_tensor = *args.dst_tensor;
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    CompTypeCvter<dtype::BFloat16, dtype::Float32> cvter(args.handle, &bundle);
    {
        cvter.src_to_comp_type(*args.src_tensor, fsrc_tensor)
                .src_to_comp_type(*args.filter_tensor, ffilter_tensor)
                .src_to_comp_type(*args.bias_tensor, fbias_tensor)
                .src_to_comp_type(*args.z_tensor, fz_tensor)
                .src_to_comp_type(*args.dst_tensor, fdst_tensor);
    }
    {
        auto convbias_opr = args.handle->create_operator<ConvBias>();
        convbias_opr->param() = args.opr->param();
        convbias_opr->param().compute_mode = Param::ComputeMode::DEFAULT;
        convbias_opr->execution_policy() = {m_impl->info()};
        convbias_opr->exec(fsrc_tensor, ffilter_tensor, fbias_tensor, fz_tensor,
                           fdst_tensor, nullptr, cvter.workspace());
    }
    { cvter.comp_to_dst_type(fdst_tensor, *args.dst_tensor); }
}

// vim: syntax=cpp.doxygen
