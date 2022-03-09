/**
 * \file src/opr/impl/search_policy/algo_chooser.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <limits>
#include <unordered_set>

#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/search_policy/algo_chooser.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/utils/invoke.h"
#include "megdnn/heuristic_cache.h"

#include "../internal/megdnn_opr_wrapper.inl"
#include "./workspace_need_limit_getter.inl"

using mgb::opr::intl::WorkspaceLimitGetter;
using namespace megdnn;
using namespace mgb;

namespace mgb {
namespace opr {

template <typename Opr>
size_t AlgoChooser<Opr>::setup_algo(
        const FixedTensorLayouts& layouts, Opr* megdnn_opr, const MGBOpr* mgb_opr,
        bool allow_weight_preprocess) {
    HeuristicCache::Key cache_key(
            megdnn_opr->handle(), megdnn_opr->get_opr_type(), layouts.data(),
            layouts.size(), &megdnn_opr->param(), sizeof(megdnn_opr->param()));
    auto rst = HeuristicCache::instance().get(cache_key);
    if (rst.policy.algo.valid()) {
        megdnn_opr->execution_policy() = rst.policy;
        return rst.workspace;
    }

    if (WorkspaceLimitGetter::is_prealloc_run(mgb_opr->owner_graph())) {
        return 0;
    }

    std::string param_str;
    Algorithm::serialize_write_pod(megdnn_opr->param(), param_str);

    auto cg = mgb_opr->owner_graph();
    rdnn::AlgoChooserDesc desc;
    desc.shared_batch_size = cg->options().fast_run_config.shared_batch_size;
    desc.binary_equal_between_batch =
            cg->options().fast_run_config.binary_equal_between_batch;
    desc.no_profiling_on_shape_change = cg->options().no_profiling_on_shape_change;
    desc.get_workspace_limit = [&](CompNode cn, size_t old_limit) {
        return WorkspaceLimitGetter::get_workspace_limit(cg, cn, old_limit);
    };

    AlgoChooserHelper helper(
            layouts, megdnn_opr, param_str, mgb_opr->comp_node(),
            mgb_opr->execution_policy(), allow_weight_preprocess, desc);

    ImplExecutionPolicy policy;
    if (auto algo_choose_hook = mgb_opr->algo_chooser()) {
        policy = algo_choose_hook(mgb_opr);
        auto strategy = rdnn::ExecutionStrategy::HEURISTIC |
                        rdnn::ExecutionStrategy::REPRODUCIBLE;
        bool retrive_from_cache = false;
        helper.construct_execution_policy(strategy, policy, retrive_from_cache);
    }
    if (!policy.algo.valid()) {
        policy = Base::get_policy(helper);
    }
    size_t workspace = helper.get_workspace_size_bytes(policy, layouts);

    std::string ret;
    ret.append(mgb_opr->dyn_typeinfo()->name);
    ret.append(": tensor layouts");
    ret += Base::format_fixlayouts(layouts);
    Algorithm* palgo = megdnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(palgo, "Unknown algo description");
    ret.append("): algo=" + std::string(palgo->name()));
    ret.append(ssprintf(
            " workspace=%.2fMiB attribute=%d", workspace / (1024 * 1024.0),
            static_cast<uint32_t>(palgo->attribute())));
    mgb_log_debug("%s", ret.c_str());

    megdnn_opr->execution_policy() = policy;

    if (mgb_opr->execution_policy().strategy & rdnn::ExecutionStrategy::HEURISTIC) {
        HeuristicCache::Result cache_result{policy, workspace};
        HeuristicCache::instance().put(cache_key, cache_result);
    }
    return workspace;
}

#define INST(Opr)                                                       \
    template size_t AlgoChooser<megdnn::Opr>::setup_algo(               \
            const FixedTensorLayouts& layouts, megdnn::Opr* megdnn_opr, \
            const MGBOpr* mgb_opr, bool allow_weight_preprocess);

MGB_FOREACH_FASTRUN_OPR(INST)
#undef INST

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
