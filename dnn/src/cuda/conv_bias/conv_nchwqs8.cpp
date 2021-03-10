/**
 * \file dnn/src/cuda/conv_bias/conv_nchwqs8.cpp
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
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/relayout_format/opr_impl.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

namespace {
inline void deduce_reformat_layout(std::unique_ptr<RelayoutFormat>& relayout,
                                   const TensorLayout& src_layout,
                                   TensorLayout& dst_layout,
                                   RelayoutFormat::Param::Mode mode,
                                   const int oc = 0, const int group = 1) {
    if (src_layout.ndim > 0) {
        RelayoutFormat::Param trans_param;
        trans_param.mode = mode;
        trans_param.oc = oc;
        trans_param.group = group;
        relayout->param() = trans_param;
        relayout->deduce_layout(src_layout, dst_layout);
    } else {
        dst_layout = src_layout;
    }
}
}  // namespace

void ConvBiasForwardImpl::AlgoFallbackNCHWQS8::make_inner_layout(
        const SizeArgs& args, TensorLayout& inner_src_layout,
        TensorLayout& inner_weight_layout, TensorLayout& inner_dst_layout,
        TensorLayout& inner_bias_layout, TensorLayout& inner_z_layout) const {
    auto relayout_src = args.handle->create_operator<RelayoutFormat>();
    deduce_reformat_layout(relayout_src, *args.src_layout, inner_src_layout,
                           RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                           args.filter_meta.group);
    deduce_reformat_layout(relayout_src, *args.filter_layout,
                           inner_weight_layout,
                           RelayoutFormat::Param::Mode::NCHW_NCHW4_WEIGHT);
    deduce_reformat_layout(relayout_src, *args.dst_layout, inner_dst_layout,
                           RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                           args.filter_meta.group);
    deduce_reformat_layout(relayout_src, *args.bias_layout, inner_bias_layout,
                           RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                           args.filter_meta.group);
    deduce_reformat_layout(relayout_src, *args.z_layout, inner_z_layout,
                           RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                           args.filter_meta.group);
};

bool ConvBiasForwardImpl::AlgoFallbackNCHWQS8::is_available(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    bool is_format_ok = param.format == param::ConvBias::Format::NCHW;
    bool is_version_ok = CUDNN_VERSION >= 7500;
    bool is_dtype_ok =
            args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS8;
    bool is_bias_ok =
            args.bias_layout->ndim == 0 ||
            (args.bias_layout->ndim == 4 && args.bias_layout->shape[0] == 1 &&
             args.bias_layout->shape[2] == 1 &&
             args.bias_layout->shape[3] == 1);
    bool is_ok = is_format_ok && is_version_ok && is_dtype_ok && is_bias_ok;
    return is_ok;
}

WorkspaceBundle ConvBiasForwardImpl::AlgoFallbackNCHWQS8::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    TensorLayout inner_src_layout;
    TensorLayout inner_weight_layout;
    TensorLayout inner_dst_layout;
    TensorLayout inner_bias_layout;
    TensorLayout inner_z_layout;
    make_inner_layout(args, inner_src_layout, inner_weight_layout,
                      inner_dst_layout, inner_bias_layout, inner_z_layout);
    auto opr = args.handle->create_operator<ConvBiasForward>();
    Param inner_conv_param = args.opr->param();
    inner_conv_param.format = Param::Format::NCHW4;
    opr->param() = inner_conv_param;
    return WorkspaceBundle(ptr, {inner_src_layout.span().dist_byte(),
                                 inner_weight_layout.span().dist_byte(),
                                 inner_dst_layout.span().dist_byte(),
                                 inner_bias_layout.span().dist_byte(),
                                 inner_z_layout.span().dist_byte(),
                                 opr->get_workspace_in_bytes(
                                         inner_src_layout, inner_weight_layout,
                                         inner_bias_layout, inner_z_layout,
                                         inner_dst_layout, nullptr)});
}

size_t ConvBiasForwardImpl::AlgoFallbackNCHWQS8::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto trans_bundle = get_workspace_bundle(nullptr, args);
    return trans_bundle.total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoFallbackNCHWQS8::exec(
        const ExecArgs& args) const {
    auto relayout_nchw_nchw4 = args.handle->create_operator<RelayoutFormat>();
    RelayoutFormat::Param in_trans;
    in_trans.mode = RelayoutFormat::Param::Mode::NCHW_NCHW4;
    in_trans.group = args.filter_meta.group;
    relayout_nchw_nchw4->param() = in_trans;

    auto relayout_weight = args.handle->create_operator<RelayoutFormat>();
    RelayoutFormat::Param weight_trans;
    weight_trans.mode = RelayoutFormat::Param::Mode::NCHW_NCHW4_WEIGHT;
    relayout_weight->param() = weight_trans;

    auto relayout_nchw4_nchw = args.handle->create_operator<RelayoutFormat>();
    RelayoutFormat::Param nchw4_nchw_trans;
    nchw4_nchw_trans.mode = RelayoutFormat::Param::Mode::NCHW4_NCHW;
    nchw4_nchw_trans.oc = args.dst_layout->shape[1];
    nchw4_nchw_trans.group = args.filter_meta.group;
    relayout_nchw4_nchw->param() = nchw4_nchw_trans;

    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    TensorLayout inner_src_layout;
    TensorLayout inner_weight_layout;
    TensorLayout inner_dst_layout;
    TensorLayout inner_bias_layout;
    TensorLayout inner_z_layout;
    make_inner_layout(args, inner_src_layout, inner_weight_layout,
                      inner_dst_layout, inner_bias_layout, inner_z_layout);
    TensorND inner_src(bundle.get(0), inner_src_layout);
    TensorND inner_weight(bundle.get(1), inner_weight_layout);
    TensorND inner_dst(bundle.get(2), inner_dst_layout);
    TensorND inner_bias(bundle.get(3), inner_bias_layout);
    TensorND inner_z(bundle.get(4), inner_z_layout);

    Param inner_conv_param = args.opr->param();
    inner_conv_param.format = Param::Format::NCHW4;
    auto inner_opr = args.handle->create_operator<ConvBiasForward>();
    inner_opr->param() = inner_conv_param;

    relayout_nchw_nchw4->exec(*args.src_tensor, inner_src, {});
    relayout_weight->exec(*args.filter_tensor, inner_weight, {});
    if (inner_bias_layout.ndim > 0) {
        relayout_nchw_nchw4->exec(*args.bias_tensor, inner_bias, {});
    }
    if (inner_z_layout.ndim > 0) {
        relayout_nchw_nchw4->exec(*args.z_tensor, inner_z, {});
    }
    inner_opr->exec(inner_src, inner_weight, inner_bias, inner_z, inner_dst,
                    nullptr, Workspace((dt_byte*)bundle.get(5), bundle.get_size(5)));
    relayout_nchw4_nchw->exec(inner_dst, *args.dst_tensor, {});
}

// vim: syntax=cpp.doxygen
