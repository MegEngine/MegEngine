#include "src/cambricon/convolution/opr_impl.h"
#include "src/cambricon/convolution/backward_data/algo.h"
#include "src/cambricon/convolution/backward_filter/algo.h"
#include "src/cambricon/convolution/forward/algo.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace cambricon {

/* ============== ConvolutionForwardImpl ============== */
void ConvolutionForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        const PreprocessedFilter* preprocessed_filter, _megdnn_workspace workspace) {
    check_exec(
            src.layout, filter.layout, dst.layout, workspace.size, preprocessed_filter);
    AlgoBase::ExecArgs args(this, src, filter, dst, workspace);
    auto algo = static_cast<ConvolutionForwardImpl::AlgoBase*>(get_algorithm_heuristic(
            src.layout, filter.layout, dst.layout, workspace.size,
            AlgoAttribute::REPRODUCIBLE, AlgoAttribute::DEFAULT));
    algo->exec(args);
}

std::vector<ConvolutionForwardImpl::Algorithm*> ConvolutionForwardImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    return megdnn::get_all_algorithms<ConvolutionForwardImpl>({this, src, filter, dst});
}

std::vector<ConvolutionForwardImpl::Algorithm*> ConvolutionForwardImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    return megdnn::get_all_algorithms_safe<ConvolutionForwardImpl>(
            {this, src, filter, dst});
}

ConvolutionForwardImpl::Algorithm* ConvolutionForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto fm = check_layout_fwd(src, filter, dst);
    AlgoBase::SizeArgs args{this, src, filter, fm, dst};
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

size_t ConvolutionForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        const PreprocessedFilter*) {
    return get_dnn_workspace(this, src, filter, dst);
}

const char* ConvolutionForwardImpl::get_algorithm_set_name() const {
    return "CONVFORWARD_DEFAULT";
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionForwardImpl)

//================== BackWardDataImpl =======================//
void ConvolutionBackwardDataImpl::exec(
        _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(filter.layout, diff.layout, grad.layout, workspace.size);
    AlgoBase::ExecArgs args(this, filter, diff, grad, workspace);
    auto algo =
            static_cast<ConvolutionBackwardDataImpl::AlgoBase*>(get_algorithm_heuristic(
                    filter.layout, diff.layout, grad.layout, workspace.size,
                    AlgoAttribute::REPRODUCIBLE, AlgoAttribute::DEFAULT));
    algo->exec(args);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*> ConvolutionBackwardDataImpl::
        get_all_algorithms(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardDataImpl>(
            {this, filter, diff, grad});
}
std::vector<ConvolutionBackwardDataImpl::Algorithm*> ConvolutionBackwardDataImpl::
        get_all_algorithms_safe(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms_safe<ConvolutionBackwardDataImpl>(
            {this, filter, diff, grad});
}

ConvolutionBackwardDataImpl::Algorithm* ConvolutionBackwardDataImpl::
        get_algorithm_heuristic(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args{this, filter, diff, grad};
    auto&& all_algos = algo_pack().all_algos;
    for (auto algo : all_algos) {
        if (algo->get_workspace_in_bytes(args) <= workspace_limit_in_bytes &&
            algo->is_available(args))
            return algo;
    }
    megdnn_assert(
            false, "No Suitable algo with src->{%s}, filter->{%s}\n",
            diff.to_string().c_str(), filter.to_string().c_str());
    return nullptr;
}

size_t ConvolutionBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    return get_dnn_workspace(this, filter, diff, grad);
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardDataImpl)

//================== BackWardFilterImpl =======================//
void ConvolutionBackwardFilterImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    AlgoBase::ExecArgs args(this, src, diff, grad, workspace);
    auto algo = static_cast<ConvolutionBackwardFilterImpl::AlgoBase*>(
            get_algorithm_heuristic(
                    src.layout, diff.layout, grad.layout, workspace.size,
                    AlgoAttribute::REPRODUCIBLE, AlgoAttribute::DEFAULT));
    algo->exec(args);
}

std::vector<ConvolutionBackwardFilterImpl::Algorithm*> ConvolutionBackwardFilterImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardFilterImpl>(
            {this, src, diff, grad});
}

std::vector<ConvolutionBackwardFilterImpl::Algorithm*> ConvolutionBackwardFilterImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms_safe<ConvolutionBackwardFilterImpl>(
            {this, src, diff, grad});
}

ConvolutionBackwardFilterImpl::Algorithm* ConvolutionBackwardFilterImpl::
        get_algorithm_heuristic(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args{this, src, diff, grad};
    auto&& all_algos = algo_pack().all_algos;
    for (auto algo : all_algos) {
        if (algo->get_workspace_in_bytes(args) <= workspace_limit_in_bytes &&
            algo->is_available(args))
            return algo;
    }
    megdnn_assert(
            false, "No Suitable algo with src->{%s}, filter->{%s}\n",
            src.to_string().c_str(), diff.to_string().c_str());
    return nullptr;
}

size_t ConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad) {
    return get_dnn_workspace(this, src, diff, grad);
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardFilterImpl)

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
