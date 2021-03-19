/**
 * \file dnn/src/cuda/convolution/forward/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/convolution/forward/algos.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/common/algo_base.h"
#include "src/common/algo_chooser.h"

using namespace megdnn;
using namespace cuda;

namespace {
std::pair<TensorLayoutArray, ConvBiasForward::Param> sub_opr_config(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst, const ConvolutionForwardImpl* opr) {
    auto conv_param = opr->param();
    DType bias_type;
    if (src.dtype.enumv() == DTypeEnum::QuantizedS8) {
        bias_type = dtype::QuantizedS32(
                src.dtype.param<dtype::QuantizedS8>().scale *

                filter.dtype.param<dtype::QuantizedS8>().scale);
    } else if (src.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        bias_type = dtype::QuantizedS32(
                src.dtype.param<dtype::Quantized8Asymm>().scale *

                filter.dtype.param<dtype::Quantized8Asymm>().scale);
    } else if (src.dtype.enumv() == DTypeEnum::Uint8 ||
               src.dtype.enumv() == DTypeEnum::Int8) {
        bias_type = dtype::Int32{};
    } else if (src.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        bias_type = dtype::QuantizedS32(
                src.dtype.param<dtype::Quantized4Asymm>().scale *

                filter.dtype.param<dtype::Quantized4Asymm>().scale);
    } else {
        megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
        bias_type = src.dtype;
    }

    std::pair<TensorLayoutArray, ConvBiasForward::Param> ret;
    ret.second = {param::ConvBias::NonlineMode::IDENTITY,
                  conv_param.mode,
                  conv_param.sparse,
                  conv_param.format,
                  conv_param.pad_h,
                  conv_param.pad_w,
                  conv_param.stride_h,
                  conv_param.stride_w,
                  conv_param.dilate_h,
                  conv_param.dilate_w,
                  conv_param.compute_mode};
    ret.first.push_back(TensorLayout({}, bias_type));
    ret.first.push_back(TensorLayout({}, dst.dtype));
    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<ConvBiasForward>> prepare_sub_opr(
        const ConvolutionForwardImpl::AlgoBase::SizeArgs& args) {
    auto conv_bias_opr = args.opr->handle()->create_operator<ConvBiasForward>();
    set_execution_policy<ConvolutionForward, ConvBiasForward*>(
            args.opr, conv_bias_opr.get());

    auto&& config = sub_opr_config(
            *args.layout_src, *args.layout_filter, *args.layout_dst,
            args.opr);
    conv_bias_opr->param() = config.second;

    return {config.first, std::move(conv_bias_opr)};
}

}  // namespace

ConvolutionForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_default);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

ConvolutionForwardImpl::AlgoPack ConvolutionForwardImpl::sm_algo_pack;

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionForwardImpl)

ConvolutionForwardImpl::AlgoBase::SizeArgs::SizeArgs(ConvolutionForwardImpl* o,
                                                     const TensorLayout& src,
                                                     const TensorLayout& filter,
                                                     const TensorLayout& dst)
        : opr{o}, layout_src{&src}, layout_filter{&filter}, layout_dst{&dst} {}

ConvolutionForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvolutionForwardImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, dst.layout),
          tensor_src{src},
          tensor_filter{filter},
          tensor_dst{dst},
          workspace{workspace} {}

std::string ConvolutionForwardImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf("src=%s, filter=%s, dst=%s",
                    layout_src->to_string().c_str(),
                    layout_filter->to_string().c_str(),
                    layout_dst->to_string().c_str());
}

/* ===================== default algo ===================== */
std::vector<Algorithm::SearchItem>
ConvolutionForwardImpl::AlgoDefault::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    auto&& config =
            sub_opr_config(layouts[0], layouts[1], layouts[2],
                           static_cast<const ConvolutionForwardImpl*>(opr));

    TensorLayoutArray conv_bias_layouts = {layouts[0], layouts[1],
                                           config.first[0], config.first[1],
                                           layouts[2]};
    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVBIAS_FORWARD, param_str,
             conv_bias_layouts}};
}

bool ConvolutionForwardImpl::AlgoDefault::is_available(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    return get_algorithm(static_cast<ConvBiasForwardImpl*>(config.second.get()),
                         *args.layout_src, *args.layout_filter, config.first[0],
                         config.first[1], *args.layout_dst);
}


size_t ConvolutionForwardImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    return config.second->get_workspace_in_bytes(
            *args.layout_src, *args.layout_filter, config.first[0],
            config.first[1], *args.layout_dst, nullptr);
}

void ConvolutionForwardImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto config = prepare_sub_opr(args);
    config.second->exec(args.tensor_src, args.tensor_filter,
                        {nullptr, config.first[0]}, {nullptr, config.first[1]},
                        args.tensor_dst, nullptr, args.workspace);
}

// vim: syntax=cpp.doxygen
