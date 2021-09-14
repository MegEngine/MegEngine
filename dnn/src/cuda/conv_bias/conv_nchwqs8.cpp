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
#include "src/cuda/relayout_format/relayout_format.h"
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

std::pair<TensorLayoutArray, ConvBiasForwardImpl::Param> sub_opr_config(
        const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
    TensorLayout inner_src_layout;
    TensorLayout inner_filter_layout;
    TensorLayout inner_bias_layout;
    TensorLayout inner_z_layout;
    TensorLayout inner_dst_layout;

    auto relayout_src = args.handle->create_operator<RelayoutFormat>();
    deduce_reformat_layout(relayout_src, *args.src_layout, inner_src_layout,
                           RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                           args.filter_meta.group);
    deduce_reformat_layout(relayout_src, *args.filter_layout,
                           inner_filter_layout,
                           RelayoutFormat::Param::Mode::NCHW_NCHW4_WEIGHT);
    bool dst_float = args.dst_layout->dtype.enumv() == DTypeEnum::Float32;
    if (dst_float) {
        inner_dst_layout = *args.dst_layout;
        inner_bias_layout = *args.bias_layout;
        inner_z_layout = *args.z_layout;
    } else {
        deduce_reformat_layout(relayout_src, *args.dst_layout, inner_dst_layout,
                               RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                               args.filter_meta.group);
        deduce_reformat_layout(relayout_src, *args.bias_layout,
                               inner_bias_layout,
                               RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                               args.filter_meta.group);
        deduce_reformat_layout(relayout_src, *args.z_layout, inner_z_layout,
                               RelayoutFormat::Param::Mode::NCHW_NCHW4, 0,
                               args.filter_meta.group);
    }

    megdnn::param::ConvBias inner_conv_param = args.opr->param();
    if (args.dst_layout->dtype.enumv() == DTypeEnum::Float32) {
        inner_conv_param.format = megdnn::param::ConvBias::Format::NCHW4_NCHW;
    } else {
        inner_conv_param.format = megdnn::param::ConvBias::Format::NCHW4;
    }
    std::pair<TensorLayoutArray, ConvBiasForwardImpl::Param> ret;
    ret.first = {inner_src_layout, inner_filter_layout, inner_bias_layout,
                 inner_z_layout, inner_dst_layout};
    ret.second = inner_conv_param;

    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<ConvBiasForward>> prepare_sub_opr(
        const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
    auto convbias_opr = args.handle->create_operator<ConvBias>();
    set_execution_policy<ConvBiasForward, ConvBiasForward*>(args.opr,
                                                            convbias_opr.get());
    auto&& config = sub_opr_config(args);
    convbias_opr->param() = config.second;

    return {config.first, std::move(convbias_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvBiasForwardImpl::AlgoFallbackNCHWQS8::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const ConvBiasForwardImpl* o = static_cast<const ConvBiasForwardImpl*>(opr);
    SizeArgs args(const_cast<ConvBiasForwardImpl*>(o), layouts[0], layouts[1],
                  layouts[2], layouts[3], layouts[4], nullptr);

    auto&& config = sub_opr_config(args);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVBIAS_FORWARD, param_str, config.first}};
}

bool ConvBiasForwardImpl::AlgoFallbackNCHWQS8::is_available(
        const SizeArgs& args) const {
    if (!args.src_layout->is_contiguous() ||
        !args.dst_layout->is_contiguous()) {
        return false;
    }
    auto&& param = args.opr->param();
    bool is_format_ok = param.format == param::ConvBias::Format::NCHW;
    bool is_version_ok = CUDNN_VERSION >= 7500;
    bool is_dtype_ok =
            (args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS8 &&
             (args.dst_layout->dtype.enumv() != DTypeEnum::QuantizedS4 ||
              args.dst_layout->dtype.enumv() != DTypeEnum::Quantized4Asymm));
    bool is_bias_ok =
            args.bias_layout->ndim == 0 ||
            (args.bias_layout->ndim == 4 && args.bias_layout->shape[0] == 1 &&
             args.bias_layout->shape[2] == 1 &&
             args.bias_layout->shape[3] == 1);
    bool is_ok = is_format_ok && is_version_ok && is_dtype_ok && is_bias_ok;
    if (!is_ok) {
        return false;
    }

    auto config = prepare_sub_opr(args);

    bool is_relayout_ok = true;
    if (args.dst_layout->dtype.enumv() != DTypeEnum::Float32) {
        is_relayout_ok = relayout_format::RelayoutFormatFast::usable(
            config.first[4], *args.dst_layout,
            RelayoutFormat::Param::Mode::NCHW4_NCHW);
    }

    return is_relayout_ok &&
           has_available_algo<ConvBiasForwardImpl>(
                   static_cast<ConvBiasForwardImpl*>(config.second.get()),
                   config.first[0], config.first[1], config.first[2],
                   config.first[3], config.first[4]);
}

WorkspaceBundle ConvBiasForwardImpl::AlgoFallbackNCHWQS8::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    size_t ws_dst = 0, ws_bias = 0, ws_z = 0;

    if (args.dst_layout->dtype.enumv() != DTypeEnum::Float32) {
        ws_bias = config.first[2].span().dist_byte();
        ws_z = config.first[3].span().dist_byte();
        ws_dst = config.first[4].span().dist_byte();
    }
    size_t inner_ws = config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2], config.first[3],
            config.first[4], nullptr);

    return WorkspaceBundle(ptr, {config.first[0].span().dist_byte(),
                                 config.first[1].span().dist_byte(), ws_bias,
                                 ws_z, ws_dst, inner_ws});
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

    auto config = prepare_sub_opr(args);
    TensorND inner_src(bundle.get(0), config.first[0]);
    TensorND inner_weight(bundle.get(1), config.first[1]);
    TensorND inner_bias(bundle.get(2), config.first[2]);
    TensorND inner_z(bundle.get(3), config.first[3]);
    TensorND inner_dst(bundle.get(4), config.first[4]);

    bool dst_float = args.dst_layout->dtype.enumv() == DTypeEnum::Float32;

    relayout_nchw_nchw4->exec(*args.src_tensor, inner_src, {});
    relayout_weight->exec(*args.filter_tensor, inner_weight, {});

    if (dst_float) {
        config.second->exec(
                inner_src, inner_weight, *args.bias_tensor, *args.z_tensor,
                *args.dst_tensor, nullptr,
                Workspace((dt_byte*)bundle.get(5), bundle.get_size(5)));
    } else {
        if (inner_bias.layout.ndim > 0) {
            relayout_nchw_nchw4->exec(*args.bias_tensor, inner_bias, {});
        }
        if (inner_z.layout.ndim > 0) {
            relayout_nchw_nchw4->exec(*args.z_tensor, inner_z, {});
        }
        config.second->exec(
                inner_src, inner_weight, inner_bias, inner_z, inner_dst,
                nullptr,
                Workspace((dt_byte*)bundle.get(5), bundle.get_size(5)));
        relayout_nchw4_nchw->exec(inner_dst, *args.dst_tensor, {});
    }
}

// vim: syntax=cpp.doxygen
