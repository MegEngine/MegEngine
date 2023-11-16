#include "src/cambricon/convolution/forward/algo.h"
#include "src/cambricon/conv_bias/algo.h"
#include "src/cambricon/conv_bias/opr_impl.h"
#include "src/cambricon/utils.h"
#include "src/common/algo_base.h"
#include "src/common/algo_chooser.h"

using namespace megdnn;
using namespace cambricon;

namespace {
std::pair<TensorLayoutArray, ConvBiasForward::Param> sub_opr_config(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        const ConvolutionForwardImpl* opr) {
    auto conv_param = opr->param();
    megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
    // todo check dtype
    DType bias_type = src.dtype;

    std::pair<TensorLayoutArray, ConvBiasForward::Param> ret;
    ret.second = {
            param::ConvBias::NonlineMode::IDENTITY,
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
    // bias cant be null
    size_t out_channel = dst.shape[3];
    ret.first.push_back(TensorLayout({1, 1, 1, out_channel}, bias_type));  // bias
    ret.first.push_back(TensorLayout({}, dst.dtype));                      // z
    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<ConvBiasForward>> prepare_sub_opr(
        const ConvolutionForwardImpl::AlgoBase::SizeArgs& args) {
    auto conv_bias_opr = args.opr->handle()->create_operator<ConvBiasForward>();
    auto&& config = sub_opr_config(
            *args.layout_src, *args.layout_filter, *args.layout_dst, args.opr);
    conv_bias_opr->param() = config.second;
    return {config.first, std::move(conv_bias_opr)};
}

WorkspaceBundle get_workspace_bundle(
        const std::pair<TensorLayoutArray, std::unique_ptr<ConvBiasForward>>&
                subopr_config,
        const ConvolutionForwardImpl::AlgoBase::SizeArgs& args) {
    size_t bias_size = subopr_config.first[0].access_bytes();
    size_t conv_work_size = subopr_config.second->get_workspace_in_bytes(
            *args.layout_src, *args.layout_filter, subopr_config.first[0],
            subopr_config.first[1], *args.layout_dst, nullptr);
    return {nullptr, {bias_size, conv_work_size}};
}

}  // namespace

ConvolutionForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_default);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

ConvolutionForwardImpl::AlgoPack ConvolutionForwardImpl::sm_algo_pack;

ConvolutionForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionForwardImpl* opr, const TensorLayout& src,
        const TensorLayout& filter, const TensorLayout& dst)
        : SizeArgs(opr, src, filter, opr->check_layout_fwd(src, filter, dst), dst) {}

ConvolutionForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionForwardImpl* opr, const TensorLayout& src,
        const TensorLayout& filter, const CanonizedFilterMeta& filter_meta,
        const TensorLayout& dst)
        : opr(opr),
          layout_src(&src),
          layout_filter(&filter),
          layout_dst(&dst),
          filter_meta(filter_meta) {}

ConvolutionForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvolutionForwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_in filter,
        _megdnn_tensor_out dst, _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, dst.layout),
          tensor_src{src},
          tensor_filter{filter},
          tensor_dst{dst},
          workspace{workspace} {}

std::string ConvolutionForwardImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf(
            "src=%s, filter=%s, dst=%s", layout_src->to_string().c_str(),
            layout_filter->to_string().c_str(), layout_dst->to_string().c_str());
}

bool ConvolutionForwardImpl::AlgoDefault::is_available(const SizeArgs& args) const {
    if (args.opr->param().format != param::Convolution::Format::NHWC) {
        return false;
    }
    auto config = prepare_sub_opr(args);
    bool valid = config.second.get()
                         ->get_algorithm_info_heuristic(
                                 *args.layout_src, *args.layout_filter, config.first[0],
                                 config.first[1], *args.layout_dst)
                         .valid();
    return valid;
}

size_t ConvolutionForwardImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    auto bundle = get_workspace_bundle(config, args);
    return bundle.total_size_in_bytes();
}

void ConvolutionForwardImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto config = prepare_sub_opr(args);
    auto bundle = get_workspace_bundle(config, args);
    bundle.set(args.workspace.raw_ptr);
    auto conv_workspace = bundle.get_workspace(1);
    cnrt_check(cnrtMemsetAsync(
            bundle.get(0), 0, bundle.get_size(0), cnrt_queue(args.opr->handle())));
    config.second->exec(
            args.tensor_src, args.tensor_filter, {bundle.get(0), config.first[0]},
            {nullptr, config.first[1]}, args.tensor_dst, nullptr, conv_workspace);
}

// vim: syntax=cpp.doxygen
