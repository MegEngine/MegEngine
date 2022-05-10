#include "src/x86/pooling/opr_impl.h"
#include "src/common/algo_chooser.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/x86/handle.h"
#include "src/x86/pooling/algo.h"
#include "src/x86/utils.h"

#if MEGDNN_X86_WITH_MKL_DNN
#include "mkldnn.hpp"
#endif

using namespace megdnn;
using namespace x86;

WorkspaceBundle megdnn::x86::get_bundle(
        const TensorLayout& src, const TensorLayout& dst, const param::Pooling& param) {
    megdnn_assert(
            is_supported(SIMDType::SSE) && src.dtype == dtype::Float32() &&
            param.format == param::Pooling::Format::NCHW &&
            param.mode == param::Pooling::Mode::MAX && param.window_h == 3 &&
            param.window_w == 3 && param.stride_h == 2 && param.stride_w == 2);
    //! max pooling 3x3 stride 2
    auto IW = src.shape[3];
    auto OW = dst.shape[3];

    WorkspaceBundle ws(
            nullptr,
            {OW * src.dtype.size(), OW * src.dtype.size(), OW * src.dtype.size(),
             (IW + 1) / 2 * src.dtype.size(), (IW + 1) / 2 * src.dtype.size()},
            16);
    return ws;
}

size_t PoolingImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    auto algo = get_algorithm(this, src, dst);
    if (!is_fallback_algo(algo)) {
        if (is_supported(SIMDType::SSE) && src.dtype == dtype::Float32() &&
            param().mode == Mode::MAX && param().format == Param::Format::NCHW &&
            param().window_h == 3 && param().window_w == 3 && param().stride_h == 2 &&
            param().stride_w == 2) {
            WorkspaceBundle ws = get_bundle(src, dst, param());

            return ws.total_size_in_bytes();
        } else {
            return 0;
        }
    } else {
        auto fallback_worksapce =
                fallback::PoolingImpl::get_workspace_in_bytes(src, dst);
        return fallback_worksapce;
    }
}
std::vector<Algorithm*> PoolingImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& dst) {
    return megdnn::get_all_algorithms<PoolingImpl>({this, src, dst});
}
std::vector<Algorithm*> PoolingImpl::get_all_algorithms_safe(
        const TensorLayout& src, const TensorLayout& dst) {
    return megdnn::get_all_algorithms_safe<PoolingImpl>({this, src, dst});
}

Algorithm* PoolingImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);

    AlgoBase::SizeArgs args(this, src, dst);
    for (auto iter : algo_pack().all_algos) {
        if (iter->is_available_attribute(args, positive_attr, negative_attr)) {
            return iter;
        }
    }
    megdnn_throw(ssprintf(
            "require algorithm with attribute(%s) and without "
            "attribute(%s), but can't get suitable algo.\n",
            Algorithm::attribute_str(positive_attr).c_str(),
            Algorithm::attribute_str(negative_attr).c_str()));
    return nullptr;
}

void PoolingImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    AlgoBase::ExecArgs args(this, src, dst, workspace);
    auto algo = get_algorithm(this, src.layout, dst.layout);
    if (!is_fallback_algo(algo)) {
        algo->exec(args);
    } else {
        fallback::PoolingImpl::exec(src, dst, workspace);
    }
}

// vim: syntax=cpp.doxygen
