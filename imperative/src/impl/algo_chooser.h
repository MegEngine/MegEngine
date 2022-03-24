#include "megbrain/rdnn/algo_chooser.h"
#include "megdnn/heuristic_cache.h"

namespace mgb {
namespace imperative {

template <typename Opr>
MGE_WIN_DECLSPEC_FUC size_t setup_algo(
        const typename mgb::rdnn::AlgoChooser<Opr>::FixedTensorLayouts& layouts,
        Opr* megdnn_opr, uint32_t shared_batch_size, bool binary_equal_between_batch,
        bool no_profiling_on_shape_change, CompNode comp_node,
        megdnn::param::ExecutionPolicy execution_policy, bool allow_weight_preprocess) {
    megdnn::HeuristicCache::Key cache_key(
            megdnn_opr->handle(), megdnn_opr->get_opr_type(), layouts.data(),
            layouts.size(), &megdnn_opr->param(), sizeof(megdnn_opr->param()));
    auto rst = megdnn::HeuristicCache::instance().get(cache_key);
    if (rst.policy.algo.valid()) {
        megdnn_opr->execution_policy() = rst.policy;
        return rst.workspace;
    }
    SmallVector<size_t> buf = rst.m_buf;
    SmallVector<char> param_buf = rst.m_param_buf;

    std::string param_str;
    megdnn::Algorithm::serialize_write_pod(megdnn_opr->param(), param_str);

    rdnn::AlgoChooserDesc desc;
    desc.shared_batch_size = shared_batch_size;
    desc.binary_equal_between_batch = binary_equal_between_batch;
    desc.no_profiling_on_shape_change = no_profiling_on_shape_change;
    desc.get_workspace_limit = [&](CompNode cn, size_t old_limit) {
        size_t free = cn.get_free_mem();
        size_t lmt = cn.get_max_block_size_available();
        return std::max(lmt, free);
    };

    using AlgoChooserHelper = typename mgb::rdnn::AlgoChooser<Opr>::AlgoChooserHelper;
    AlgoChooserHelper helper(
            layouts, megdnn_opr, param_str, comp_node, execution_policy,
            allow_weight_preprocess, desc);

    megdnn::ExecutionPolicy policy;
    policy = mgb::rdnn::AlgoChooser<Opr>::get_policy(helper);
    size_t workspace = helper.get_workspace_size_bytes(policy, layouts);
    megdnn_opr->execution_policy() = policy;

    if (execution_policy.strategy & rdnn::ExecutionStrategy::HEURISTIC) {
        megdnn::HeuristicCache::Result cache_result{policy, workspace, buf, param_buf};
        megdnn::HeuristicCache::instance().put(cache_key, cache_result);
    }
    return workspace;
}

}  // namespace imperative
}  // namespace mgb
