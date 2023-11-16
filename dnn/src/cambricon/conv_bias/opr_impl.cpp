#include "src/cambricon/conv_bias/opr_impl.h"
#include "src/cambricon/conv_bias/algo.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace cambricon {

void ConvBiasForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
        _megdnn_tensor_in z, _megdnn_tensor_out dst,
        const PreprocessedFilter* preprocessed_filter, _megdnn_workspace workspace) {
    check_exec_allow_noncontiguous(
            src.layout, filter.layout, bias.layout, z.layout, dst.layout,
            workspace.size, preprocessed_filter);
    AlgoBase::ExecArgs args(this, src, filter, bias, z, dst, workspace);
    auto algo = static_cast<ConvBiasForwardImpl::AlgoBase*>(get_algorithm_heuristic(
            src.layout, filter.layout, bias.layout, z.layout, dst.layout,
            workspace.size, AlgoAttribute::DEFAULT, AlgoAttribute::DEFAULT));
    algo->exec(args);
}

std::vector<ConvBiasForward::Algorithm*> ConvBiasForwardImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst) {
    return megdnn::get_all_algorithms<ConvBiasForwardImpl>(
            {this, src, filter, bias, z, dst});
}

std::vector<ConvBiasForward::Algorithm*> ConvBiasForwardImpl::get_all_algorithms_safe(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst) {
    return megdnn::get_all_algorithms_safe<ConvBiasForwardImpl>(
            {this, src, filter, bias, z, dst});
}

ConvBiasForward::Algorithm* ConvBiasForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr) {
    auto fm = check_layout_fwd(src, filter, dst);
    AlgoBase::SizeArgs args{this, src, filter, fm, bias, z, dst};
    auto&& all_algos = algo_pack().all_algos;
    for (auto algo : all_algos) {
        if (algo->get_workspace_in_bytes(args) <= workspace_limit_in_bytes &&
            algo->is_available(args))
            return algo;
    }
    megdnn_assert(
            false, "No Suitable algo with src->{%s}, filter->{%s}\n",
            src.to_string().c_str(), filter.to_string().c_str());
    return nullptr;
}

size_t ConvBiasForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst, const PreprocessedFilter*) {
    return get_dnn_workspace(this, src, filter, bias, z, dst);
}

const char* ConvBiasForwardImpl::get_algorithm_set_name() const {
    return "ConvBiasDEFAULT";
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvBiasForwardImpl)

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
