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

#include "megbrain/opr/search_policy/algo_chooser.h"
#include <limits>
#include <unordered_set>
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/opr/search_policy/profiler.h"

#include "../internal/invoke.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "./workspace_need_limit_getter.inl"

//! TODO: here has to be know some megdnn::opr when there is produced midout.h
//! fix it if there is another graceful way.
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "megdnn/oprs/base.h"
#include "midout.h"
MIDOUT_DECL(megbrain_opr_algo_chooser)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_algo_chooser, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using mgb::opr::intl::WorkspaceLimitGetter;
using namespace megdnn;
using namespace mgb;

#define APPLY(statement, ...)                                  \
    mgb::apply([&](const auto&... args) { return statement; }, \
               std::tuple_cat(__VA_ARGS__))

// timeout delta to be added with fastest known algorithm for new algos
constexpr double TIMEOUT_TOLERANCE = 2;

#define CACHE_KEY_VERSION "v4"

namespace {
template <typename Opr>
std::string profile_name(Opr* opr) {
    std::string ret =
            std::string(MegDNNOpr2MGBOpr<Opr>::MGBOpr::typeinfo()->name) +
            CACHE_KEY_VERSION;
    ret.append(opr->get_algorithm_set_name());
    return ret;
}

template <typename Opr>
std::string format_fixlayouts(
        const typename opr::AlgoChooser<Opr>::FixedTensorLayouts& layouts,
        size_t arity_in, size_t arity_out) {
    std::string ret;
    ret.append(": tensor layouts(");
    for (size_t i = 0; i < arity_in; ++i) {
        if (i) {
            ret.append(", ");
        }
        ret.append(layouts[i].to_string() + " ");
        ret.append(layouts[i].dtype.name());
    }
    ret.append(") -> (");
    for (size_t i = 0; i < arity_out; ++i) {
        if (i) {
            ret.append(", ");
        }
        ret.append(layouts[i + arity_in].to_string() + " ");
        ret.append(layouts[i + arity_in].dtype.name());
    }
    return ret;
}

/**
 * \brief Check if the sub opr list has circular dependence.
 */
class CircularDepsChecker {
    struct SearchItemStorage {
        std::string data_hold;
        size_t hash = 0;

        SearchItemStorage(const Algorithm::SearchItem& item) {
            Algorithm::serialize_write_pod(item.opr_type, data_hold);
            for (auto&& layout : item.layouts) {
                data_hold += layout.serialize();
            }
            data_hold += item.param;
        }

        SearchItemStorage& init_hash() {
            hash = XXHash64CT::hash(data_hold.data(), data_hold.size(),
                                    20201225);
            return *this;
        }

        bool operator==(const SearchItemStorage& rhs) const {
            return data_hold == rhs.data_hold;
        }

        struct Hash {
            size_t operator()(const SearchItemStorage& s) const {
                return s.hash;
            }
        };
    };
    std::unordered_set<SearchItemStorage, SearchItemStorage::Hash> m_set;

public:
    void put(const megdnn::Algorithm::SearchItem& key) {
        SearchItemStorage key_storage(key);
        key_storage.init_hash();
        mgb_assert(m_set.find(key_storage) == m_set.end(),
                   "Circular dependency during flatten search space");
        auto ret = m_set.insert(std::move(key_storage));
        mgb_assert(ret.second);
    }
    void remove(const megdnn::Algorithm::SearchItem& key) {
        SearchItemStorage key_storage(key);
        key_storage.init_hash();
        auto&& iter = m_set.find(key_storage);
        mgb_assert(iter != m_set.end());
        m_set.erase(iter);
    }
};

///////////////// OprTypeTrait /////////////////////////////
template <megdnn::Algorithm::OprType>
struct OprFromOprTypeTrait;

template <typename Opr>
struct OprTypeFromOprTrait;

#define cb(_opr_type, _opr)                                             \
    template <>                                                         \
    struct OprFromOprTypeTrait<megdnn::Algorithm::OprType::_opr_type> { \
        using Opr = megdnn::_opr;                                       \
    };                                                                  \
    template <>                                                         \
    struct OprTypeFromOprTrait<megdnn::_opr> {                          \
        constexpr static megdnn::Algorithm::OprType opr_type =          \
                megdnn::Algorithm::OprType::_opr_type;                  \
    }

cb(MATRIX_MUL_FORWARD, MatrixMulForward);
cb(BATCHED_MATRIX_MUL_FORWARD, BatchedMatrixMulForward);
cb(CONVOLUTION_FORWARD, ConvolutionForward);
cb(CONVOLUTION_BACKWARD_DATA, ConvolutionBackwardData);
cb(CONVOLUTION_BACKWARD_FILTER, ConvolutionBackwardFilter);
cb(CONVOLUTION3D_FORWARD, Convolution3DForward);
cb(CONVOLUTION3D_BACKWARD_DATA, Convolution3DBackwardData);
cb(CONVOLUTION3D_BACKWARD_FILTER, Convolution3DBackwardFilter);
cb(LOCAL_SHARE_FORWARD, LocalShareForward);
cb(LOCAL_SHARE_BACKWARD_DATA, LocalShareBackwardData);
cb(LOCAL_SHARE_BACKWARD_FILTER, LocalShareBackwardFilter);
cb(DEFORMABLE_CONV_FORWARD, DeformableConvForward);
cb(DEFORMABLE_CONV_BACKWARD_DATA, DeformableConvBackwardData);
cb(DEFORMABLE_CONV_BACKWARD_FILTER, DeformableConvBackwardFilter);
cb(BATCH_CONV_FORWARD, BatchConvBiasForward);
cb(CONVBIAS_FORWARD, ConvBiasForward);

#undef cb

// clang-format off
#define FOREACH_OPR_TYPE_WITH_STMT(cb, stmt)  \
    cb(MATRIX_MUL_FORWARD, stmt)              \
    cb(BATCHED_MATRIX_MUL_FORWARD, stmt)      \
    cb(CONVOLUTION_FORWARD, stmt)             \
    cb(CONVOLUTION_BACKWARD_DATA, stmt)       \
    cb(CONVOLUTION_BACKWARD_FILTER, stmt)     \
    cb(CONVOLUTION3D_FORWARD, stmt)           \
    cb(CONVOLUTION3D_BACKWARD_DATA, stmt)     \
    cb(CONVOLUTION3D_BACKWARD_FILTER, stmt)   \
    cb(LOCAL_SHARE_FORWARD, stmt)             \
    cb(LOCAL_SHARE_BACKWARD_DATA, stmt)       \
    cb(LOCAL_SHARE_BACKWARD_FILTER, stmt)     \
    cb(DEFORMABLE_CONV_FORWARD, stmt)         \
    cb(DEFORMABLE_CONV_BACKWARD_DATA, stmt)   \
    cb(DEFORMABLE_CONV_BACKWARD_FILTER, stmt) \
    cb(BATCH_CONV_FORWARD, stmt)              \
    cb(CONVBIAS_FORWARD, stmt)
// clang-format on

#define _OPR_TYPE_CASE(_opr_type, _stmt)             \
    case Algorithm::OprType::_opr_type: {            \
        using _Opr = typename OprFromOprTypeTrait<   \
                Algorithm::OprType::_opr_type>::Opr; \
        _stmt;                                       \
        break;                                       \
    }

#define FOREACH_OPR_TYPE_DISPATCH(_search_items, _stmt)          \
    for (size_t _item_idx = 0; _item_idx < _search_items.size(); \
         _item_idx++) {                                          \
        auto&& _item = _search_items[_item_idx];                 \
        switch (_item.opr_type) {                                \
            FOREACH_OPR_TYPE_WITH_STMT(_OPR_TYPE_CASE, _stmt)    \
            default:                                             \
                mgb_throw(MegBrainError, "unknown opr_type");    \
        }                                                        \
    }

template <typename Opr>
TensorLayoutArray to_layout_array(
        const typename opr::AlgoChooser<Opr>::FixedTensorLayouts& layouts) {
    TensorLayoutArray ret;
    for (auto&& layout : layouts) {
        ret.push_back(layout);
    }
    return ret;
}

template <typename Opr>
typename opr::AlgoChooser<Opr>::FixedTensorLayouts to_fixed_layouts(
        const TensorLayoutArray& layouts) {
    typename opr::AlgoChooser<Opr>::FixedTensorLayouts ret;
    mgb_assert(ret.size() == layouts.size());
    size_t idx = 0;
    for (auto&& layout : layouts) {
        ret[idx++] = layout;
    }
    return ret;
}

/**
 * flatten search space in postorder traversal
 * The subopr search construct a search tree
 *
 *           A
 *        /    \
 *       B1B2   C
 *      /     \
 *     D1D2D3   E
 * We use postorder traverse the search tree.
 * D1 -> D2 -> D3 -> E -> B1 -> B2 -> C -> A
 */
template <typename Opr>
std::vector<megdnn::Algorithm::SearchItem> flatten_search_space(
        const typename opr::AlgoChooser<Opr>::ExeContext& ctx,
        CircularDepsChecker& checker) {
    auto&& search_item = megdnn::Algorithm::SearchItem{
            OprTypeFromOprTrait<Opr>::opr_type, ctx.param(),
            to_layout_array<Opr>(ctx.layouts())};
    checker.put(search_item);
    std::vector<megdnn::Algorithm::SearchItem> ret;
    for (auto algo_info : ctx.get_all_candidates()) {
        megdnn::Algorithm* algo = ctx.get_algorithm_from_desc(algo_info.desc);
        mgb_assert(algo, "Unknown algo description");
        std::vector<megdnn::Algorithm::SearchItem>&& sub_items =
                algo->get_subopr_list(to_layout_array<Opr>(ctx.layouts()),
                                      ctx.megdnn_opr());

        FOREACH_OPR_TYPE_DISPATCH(sub_items, {
            auto&& megdnn_opr =
                    opr::intl::create_megdnn_opr<_Opr>(ctx.comp_node());
            megdnn_opr->param() =
                    Algorithm::deserialize_read_pod<typename _Opr::Param>(
                            _item.param);
            typename opr::AlgoChooser<_Opr>::ExeContext sub_ctx(
                    to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                    _item.param, ctx.mgb_opr(), ctx.comp_node(),
                    ctx.execution_policy(), ctx.allow_weight_preprocess());
            auto space = flatten_search_space<_Opr>(sub_ctx, checker);
            ret.insert(ret.end(), space.begin(), space.end());
        });
    }
    ret.push_back(search_item);
    checker.remove(search_item);
    return ret;
}

//! return pair<positive_attr, negative_attr>
std::pair<AlgoAttribute, AlgoAttribute>
extract_algo_attribute_from_execution_strategy(
        const ExecutionStrategy& strategy) {
    std::pair<AlgoAttribute, AlgoAttribute> ret =
            std::make_pair(AlgoAttribute::DEFAULT, AlgoAttribute::DEFAULT);
    if (strategy & ExecutionStrategy::REPRODUCIBLE) {
        ret.first |= AlgoAttribute::REPRODUCIBLE;
    }
    if (strategy & ExecutionStrategy::OPTIMIZED) {
        ret.second |= AlgoAttribute::NAIVE;
    }
    return ret;
}

}  // namespace

namespace mgb {
namespace opr {

template <typename Opr>
void AlgoChooser<Opr>::profile(ExeContext& ctx,
                               ExecutionStrategy selected_strategy) {
    if (ctx.get_profile_result_from_cache(selected_strategy).valid())
        return;
    AlgoChooserProfileCache::Result prof_rst;

    auto target_attr =
            extract_algo_attribute_from_execution_strategy(selected_strategy);
    std::string layouts_str = format_fixlayouts<Opr>(ctx.layouts(), arity_in, arity_out);
    double cur_timeout = 0;

    auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
            ctx.owner_graph(), ctx.comp_node(),
            ctx.execution_policy().workspace_limit);
    RealTimer timer;
    for (auto algo : ctx.get_all_candidates()) {
        Maybe<AlgoChooserProfileCache::ResultEntry> cur_rst;
        std::string msg = ssprintf("profiling %s algorithm %s %s",
                                   ctx.mgb_opr()->dyn_typeinfo()->name,
                                   algo.desc.name.c_str(), layouts_str.c_str());
        ImplExecutionPolicy policy;
        policy.algo = algo.desc;
        ctx.construct_execution_policy(selected_strategy, policy);
        if (ctx.get_workspace_size_bytes(policy) >= workspace_limit) {
            continue;
        }
        auto palgo = ctx.megdnn_opr()->get_algorithm_from_desc(policy.algo);
        if (!(palgo->contain_attribute_all(target_attr.first) &&
              !palgo->contain_attribute_any(target_attr.second))) {
            mgb_log_debug(
                    "skip algo %s with attribute(%s), which is not match the "
                    "profile strategy required contain attribute(%s) and not "
                    "contain attribute(%s).",
                    algo.desc.name.c_str(),
                    Algorithm::attribute_str(palgo->attribute()).c_str(),
                    Algorithm::attribute_str(target_attr.first).c_str(),
                    Algorithm::attribute_str(target_attr.second).c_str());
            continue;
        }

        timer.reset();
        MGB_TRY { cur_rst = ctx.profile_single_algo(policy, cur_timeout); }
        MGB_CATCH(std::exception & exc, {
            mgb_log_warn("caught exception during %s: %s", msg.c_str(),
                         exc.what());
            continue;
        })
        MGB_CATCH(..., {
            mgb_log_warn("caught exception during %s", msg.c_str());
            continue;
        })
        if (!cur_rst.valid()) {
            mgb_log_warn("timeout when %s; timeout setting: %.3fsec",
                         msg.c_str(), cur_timeout);
            continue;
        }
        if (!cur_timeout) {
            cur_timeout = timer.get_secs() + TIMEOUT_TOLERANCE;
        } else {
            cur_timeout =
                    std::min(cur_timeout, timer.get_secs() + TIMEOUT_TOLERANCE);
        }
        auto&& rst = cur_rst.val();
        mgb_log_debug("%s: workspace: %zu; time: %.3gsec", msg.c_str(),
                      rst.workspace, rst.time);
        prof_rst.push_back(rst);
    }
    std::string msg = ssprintf(
            "no usable %s algorithm %s with attribute(%s) and without "
            "attribute(%s)",
            ctx.mgb_opr()->dyn_typeinfo()->name, layouts_str.c_str(),
            Algorithm::attribute_str(target_attr.first).c_str(),
            Algorithm::attribute_str(target_attr.second).c_str());
    mgb_assert(!prof_rst.empty(), "%s", msg.c_str());

    FixedTensorLayouts origin_layouts = ctx.layouts();
    typename Opr::Param origin_param = ctx.megdnn_opr()->param();
    AlgoChooserProfileCache::Key cache_key{origin_layouts.data(),
                                           origin_layouts.size(), &origin_param,
                                           sizeof(origin_param)};

    AlgoChooserProfileCache cache(ctx.comp_node(),
                                  profile_name(ctx.megdnn_opr()).c_str());
    cache.put(cache_key, prof_rst);
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplExecutionPolicy
AlgoChooser<Opr>::choose_by_profile(ExeContext& ctx,
                                    ExecutionStrategy selected_strategy,
                                    bool enable_update) {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("AlgoChooser::choose_by_profile")))
    if (ctx.owner_graph()->options().no_profiling_on_shape_change) {
        auto policy = ctx.megdnn_opr()->execution_policy();
        if (policy.algo.valid()){
            return policy;
        }
        if (!algo_usable_on_shape_change<Opr>()) {
            mgb_log_warn(
                    "choose algo by heuristic, which may cause performance "
                    "regression.");
            return ctx.choose_by_heuristic(selected_strategy);
        }
    }

    if (enable_update) {
        CircularDepsChecker circular_deps_checker;
        auto&& search_items =
                flatten_search_space<Opr>(ctx, circular_deps_checker);
        FOREACH_OPR_TYPE_DISPATCH(search_items, {
            auto&& megdnn_opr = intl::create_megdnn_opr<_Opr>(ctx.comp_node());
            megdnn_opr->param() =
                    Algorithm::deserialize_read_pod<typename _Opr::Param>(
                            _item.param);
            typename AlgoChooser<_Opr>::ExeContext sub_ctx(
                    to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                    _item.param, ctx.mgb_opr(), ctx.comp_node(),
                    ctx.execution_policy(), ctx.allow_weight_preprocess());
            AlgoChooser<_Opr>::profile(sub_ctx, selected_strategy);
        });
    }
    typename AlgoChooser<Opr>::ImplExecutionPolicy policy;
    ctx.construct_execution_policy(selected_strategy, policy);
    return policy;
    MIDOUT_E
}

template <typename Opr>
size_t AlgoChooser<Opr>::setup_algo(const FixedTensorLayouts& layouts,
                                    Opr* megdnn_opr, const MGBOpr* mgb_opr,
                                    bool allow_weight_preprocess) {
    if (WorkspaceLimitGetter::is_prealloc_run(mgb_opr->owner_graph())) {
        return 0;
    }

    std::string param_str;
    Algorithm::serialize_write_pod(megdnn_opr->param(), param_str);
    ExeContext ctx(layouts, megdnn_opr, param_str, mgb_opr,
                   mgb_opr->comp_node(), mgb_opr->execution_policy(),
                   allow_weight_preprocess);

    ImplExecutionPolicy policy;
    if (auto algo_choose_hook = mgb_opr->algo_chooser()) {
        policy = algo_choose_hook(mgb_opr);
        ctx.construct_execution_policy((ExecutionStrategy::HEURISTIC |
                                        ExecutionStrategy::REPRODUCIBLE),
                                       policy, false);
    }
    if (!policy.algo.valid()) {
        policy = get_policy(ctx);
    }
    size_t workspace = ctx.get_workspace_size_bytes(policy);

    std::string ret;
    ret.append(mgb_opr->dyn_typeinfo()->name);
    ret += format_fixlayouts<Opr>(layouts, arity_in, arity_out);
    Algorithm* palgo = megdnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(palgo, "Unknown algo description");
    ret.append("): algo=" + std::string(palgo->name()));
    ret.append(ssprintf(" workspace=%.2fMiB attirbute(%s)",
                        workspace / (1024 * 1024.0),
                        Algorithm::attribute_str(palgo->attribute()).c_str()));
    mgb_log_debug("%s", ret.c_str());

    megdnn_opr->execution_policy() = policy;
    return workspace;
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplExecutionPolicy AlgoChooser<Opr>::get_policy(
        ExeContext& ctx) {
    MGB_MARK_USED_VAR(TIMEOUT_TOLERANCE);
    auto opr_strategy = ctx.execution_policy().strategy;
    if ((opr_strategy & ExecutionStrategy::HEURISTIC) &&
               (opr_strategy & ExecutionStrategy::PROFILE)) {
        ImplExecutionPolicy policy =
                choose_by_profile(ctx, opr_strategy, false);
        if (!policy.algo.valid())
            policy = ctx.choose_by_heuristic(opr_strategy);
        return policy;
    } else if (!static_cast<int>(opr_strategy) ||
               (opr_strategy & ExecutionStrategy::HEURISTIC)) {
        return ctx.choose_by_heuristic(opr_strategy);
    }
#if MGB_ENABLE_FASTRUN
    else if (opr_strategy & ExecutionStrategy::PROFILE) {
        return choose_by_profile(ctx, opr_strategy);
    }
#endif
    else {
        mgb_throw(GraphError, "bad ExecutionPolicy strategy");
    }
}

#define INST(Opr)                                                       \
    template AlgoChooser<megdnn::Opr>::ImplExecutionPolicy              \
    AlgoChooser<megdnn::Opr>::get_policy(ExeContext& ctx);              \
    template void AlgoChooser<megdnn::Opr>::profile(ExeContext& ctx,    \
                                                    ExecutionStrategy); \
    template AlgoChooser<megdnn::Opr>::ImplExecutionPolicy              \
    AlgoChooser<megdnn::Opr>::choose_by_profile(                        \
            ExeContext& ctx, ExecutionStrategy, bool enable_update);    \
    template size_t AlgoChooser<megdnn::Opr>::setup_algo(               \
            const FixedTensorLayouts& layouts, megdnn::Opr* megdnn_opr, \
            const MGBOpr* mgb_opr, bool allow_weight_preprocess);

MGB_FOREACH_FASTRUN_OPR(INST)

#undef INST

//////////////////////////////// ExeContext /////////////////////////////
template <typename Opr>
AlgoChooser<Opr>::ExeContext::ExeContext(
        const FixedTensorLayouts& layouts, Opr* megdnn_opr,
        const std::string& param_str, const cg::OperatorNodeBase* mgb_opr,
        const CompNode& cn,
        const megdnn::param::ExecutionPolicy& execution_policy,
        bool allow_weight_preprocess)
        : m_layouts{layouts},
          m_megdnn_opr{megdnn_opr},
          m_param{param_str},
          m_base_mgb_opr{mgb_opr},
          m_cn{cn},
          m_execution_policy{execution_policy},
          m_allow_weight_preprocess{allow_weight_preprocess} {
    mgb_assert(m_layouts.size() == layouts.size());
    static_assert(std::tuple_size<FixedTensorLayouts>::value == 3 ||
                          std::tuple_size<FixedTensorLayouts>::value == 5 ||
                          std::tuple_size<FixedTensorLayouts>::value == 8,
                  "Convolution AlgoChooser assumes arity = 3 , 5 or 8 (for "
                  "deformable conv)");
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplAlgo
AlgoChooser<Opr>::ExeContext::get_profile_result_from_cache(
        ExecutionStrategy selected_strategy) const {
    MIDOUT_B(Opr,
             midout_iv(MGB_HASH_STR(
                     "AlgoChooser::ExeContext::get_profile_result_from_cache")))
    AlgoChooserProfileCache cache(m_cn,
                                  profile_name(m_megdnn_opr).c_str());

    typename Opr::Param origin_param = m_megdnn_opr->param();
    AlgoChooserProfileCache::Key cache_key{m_layouts.data(), m_layouts.size(),
                                           &origin_param, sizeof(origin_param)};
    auto&& rst = cache.get(cache_key);
    if (!rst.valid())
        return {};

    auto&& prof = rst.val();
    std::unordered_map<std::string, ImplAlgo> algo_map;
    for (auto i : get_all_candidates()) {
        auto ins = algo_map.emplace(i.desc.name.c_str(), i);
        mgb_assert(ins.second, "duplicated algo name: %s", i.desc.name.c_str());
    }

    if (prof.empty())
        return {};
    for (auto&& i : prof) {
        if (!(selected_strategy & ExecutionStrategy::REPRODUCIBLE) ||
            static_cast<AlgoAttribute>(i.attribute) &
                    AlgoAttribute::REPRODUCIBLE) {
            auto iter = algo_map.find(i.algo);
            mgb_assert(iter != algo_map.end(),
                       "algorithm %s exists in "
                       "profiling result but not in algo_map; please "
                       "report this "
                       "bug; opr: %s{%s}, layouts: %s ",
                       i.algo.c_str(), m_base_mgb_opr->cname(),
                       m_base_mgb_opr->dyn_typeinfo()->name,
                       format_fixlayouts<Opr>(m_layouts, arity_in, arity_out)
                               .c_str());
            return iter->second;
        }
    }

    mgb_log_error(
            "Workspace requirement (%zu) could not be satisfied. Abort now "
            "to "
            "avoid further problems",
            WorkspaceLimitGetter::get_workspace_limit(
                    m_base_mgb_opr->owner_graph(), m_cn,
                    m_execution_policy.workspace_limit));
    mgb_trap();
    MIDOUT_E
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplExecutionPolicy
AlgoChooser<Opr>::ExeContext::choose_by_heuristic(
        ExecutionStrategy selected_strategy) const {
    if (m_execution_policy.workspace_limit !=
        std::numeric_limits<decltype(
                m_execution_policy.workspace_limit)>::max()) {
        mgb_log_warn(
                "workspace_limit should not be setted if choose algo by "
                "heuristic");
    }
    auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
            owner_graph(), m_cn, m_execution_policy.workspace_limit);
    auto attr =
            extract_algo_attribute_from_execution_strategy(selected_strategy);
    ImplExecutionPolicy policy;
    policy.algo =
            APPLY(m_megdnn_opr->get_algorithm_info_heuristic(
                          args..., workspace_limit, attr.first, attr.second),
                  m_layouts)
                    .desc;

    Algorithm* algo = m_megdnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(algo, "Unknown algo description");
    std::vector<Algorithm::SearchItem>&& sub_items = algo->get_subopr_list(
            to_layout_array<Opr>(m_layouts), m_megdnn_opr);

    FOREACH_OPR_TYPE_DISPATCH(sub_items, {
        auto&& megdnn_opr = intl::create_megdnn_opr<_Opr>(m_cn);
        megdnn_opr->param() =
                Algorithm::deserialize_read_pod<typename _Opr::Param>(
                        _item.param);
        typename AlgoChooser<_Opr>::ExeContext sub_ctx(
                to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                _item.param, m_base_mgb_opr, m_cn, m_execution_policy,
                m_allow_weight_preprocess);
        policy.sub_policy.push_back(
                sub_ctx.choose_by_heuristic(selected_strategy));
    });

    return policy;
}

template <typename Opr>
std::vector<typename AlgoChooser<Opr>::ImplAlgo>
AlgoChooser<Opr>::ExeContext::get_all_candidates() const {
    auto heu = choose_by_heuristic(ExecutionStrategy::HEURISTIC);
    auto&& ret = APPLY(m_megdnn_opr->get_all_algorithms_info(args...), m_layouts);
    bool found = false;
    for (size_t i = 0; i < ret.size(); ++i) {
        if (ret[i].desc == heu.algo) {
            found = true;
            std::swap(ret[i], ret[0]);
            break;
        }
    }

    Algorithm* palgo = m_megdnn_opr->get_algorithm_from_desc(heu.algo);
    mgb_assert(palgo, "Unknown algo description");
    mgb_assert(found,
               "algo %s got by heuristic not found in "
               "candidate list",
               palgo->name());
    return std::move(ret);
}

template <typename Opr>
void AlgoChooser<Opr>::ExeContext::construct_execution_policy(
        ExecutionStrategy selected_strategy,
        typename AlgoChooser<Opr>::ImplExecutionPolicy& policy,
        bool retrive_from_cache) const {
    if (!policy.algo.valid()) {
        if (retrive_from_cache) {
            policy.algo =
                    get_profile_result_from_cache(selected_strategy).desc;
        } else {
            auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
                    owner_graph(), m_cn, m_execution_policy.workspace_limit);

            auto attr = extract_algo_attribute_from_execution_strategy(
                    selected_strategy);
            policy.algo = APPLY(m_megdnn_opr->get_algorithm_info_heuristic(
                                        args..., workspace_limit, attr.first,
                                        attr.second),
                                m_layouts)
                                  .desc;
        }
        mgb_assert(policy.algo.valid(),
                   "No algo found from cache or heuristic, maybe some error "
                   "occured");
    }

    Algorithm* algo = m_megdnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(algo, "Unknown algo description");
    std::vector<Algorithm::SearchItem>&& sub_items = algo->get_subopr_list(
            to_layout_array<Opr>(m_layouts), m_megdnn_opr);

    FOREACH_OPR_TYPE_DISPATCH(sub_items, {
        auto&& megdnn_opr = intl::create_megdnn_opr<_Opr>(m_cn);
        megdnn_opr->param() =
                Algorithm::deserialize_read_pod<typename _Opr::Param>(
                        _item.param);
        typename AlgoChooser<_Opr>::ExeContext sub_ctx(
                to_fixed_layouts<_Opr>(_item.layouts), megdnn_opr.get(),
                _item.param, m_base_mgb_opr, m_cn, m_execution_policy,
                m_allow_weight_preprocess);
        policy.sub_policy.push_back({});
        sub_ctx.construct_execution_policy(selected_strategy,
                                           policy.sub_policy.back(),
                                           retrive_from_cache);
    });

    return;
}

template <typename Opr>
size_t AlgoChooser<Opr>::ExeContext::get_workspace_size_bytes(
        const ImplExecutionPolicy& policy) const {
    m_megdnn_opr->execution_policy() = policy;
    size_t result;
    if_constexpr<opr_supports_preprocess<Opr>()>(
            [&](auto _) {
                auto&& opr = _(m_megdnn_opr);
                auto prep = this->construct_fake_preprocess_filter();
                PreprocessFilter<Opr>* prep_ptr =
                        prep.valid() ? &prep.val() : nullptr;
                result = std::max(
                        APPLY(opr->get_preprocess_workspace_in_bytes(args...),
                              m_layouts),
                        APPLY(opr->get_workspace_in_bytes(args..., prep_ptr),
                              m_layouts));
            },
            /* else */
            [&](auto _) {
                result = APPLY(_(m_megdnn_opr)->get_workspace_in_bytes(args...),
                               m_layouts);
            });
    return result;
}

template <typename Opr>
Maybe<AlgoChooserProfileCache::ResultEntry>
AlgoChooser<Opr>::ExeContext::profile_single_algo(
        const ImplExecutionPolicy& policy, double& timeout) const {
    typename TimedProfiler<Opr>::Param param;
    // force check copy size <= dest len-1 from gcc8 for safe
    param.execution_policy =
            TimedProfiler<Opr>::Param::ExecutionPolicyBlob::serialize(policy);
    param.workspace = get_workspace_size_bytes(policy);
    for (int i = 0; i < arity; ++i) {
        auto&& src = m_layouts[i];
        mgb_assert(src.format.is_default() &&
                           (src.dtype.category() == DTypeCategory::FLOAT ||
                            src.dtype.category() == DTypeCategory::INT ||
                            src.dtype.category() == DTypeCategory::QUANTIZED),
                   "unsupported layout in profiling: %s",
                   src.to_string().c_str());
        param.dtypes[i] = src.dtype.enumv();
    }
    param.comp_node_loc = m_cn.locator();
    mgb_assert(param.shapes.size() == m_layouts.size());
    for (size_t i = 0; i < param.shapes.size(); ++i)
        param.shapes[i] = m_layouts[i];
    param.opr_param = m_megdnn_opr->param();
    param.allow_weight_preprocess = m_allow_weight_preprocess;

    Algorithm* palgo = m_megdnn_opr->get_algorithm_from_desc(policy.algo);
    mgb_assert(palgo, "Unknown algo description");
    auto rst = TimedProfiler<Opr>::profile(param, timeout);
    // MIOpen conv profiles all available algos when a specfic shape is
    // provided for the first time, which probably adds to the result time.
    // Therefore, a second profile execution is needed.
    if (strncmp(palgo->name(), "MIOpen", 6) == 0)
        rst = TimedProfiler<Opr>::profile(param, timeout);
    if (!rst.valid())
        return None;
    return AlgoChooserProfileCache::ResultEntry{
            palgo->name(),
            static_cast<uint32_t>(palgo->attribute()),
            rst.val().time, param.workspace};
}

template <typename Opr>
Maybe<PreprocessFilter<Opr>>
AlgoChooser<Opr>::ExeContext::construct_fake_preprocess_filter() const {
    Maybe<PreprocessFilter<Opr>> result = None;
    if_constexpr<opr_supports_preprocess<Opr>()>([&](auto _) {
        if (!m_allow_weight_preprocess)
            return;
        auto opr = _(m_megdnn_opr);
        auto layouts = APPLY(opr->deduce_preprocessed_filter_layout(args...),
                             m_layouts);
        //! No preprocess layout means no need weight preprocess
        if (layouts.empty()) {
            return;
        }
        //! all layouts arm empty means no need weight preprocess
        bool layout_valid = false;
        for (auto&& layout : layouts) {
            if (!layout.is_empty()) {
                layout_valid = true;
            }
        }
        if (!layout_valid) {
            return;
        }

        result = PreprocessFilter<Opr>{};
        auto& res = result.val();
        res.algorithm_id = nullptr;
        res.tensors.resize(layouts.size());
        for (size_t i = 0; i < layouts.size(); i++) {
            res.tensors[i] = megdnn::TensorND(nullptr, layouts[i]);
        }
    });
    return result;
}

#define INST(Opr)                                                              \
    template AlgoChooser<megdnn::Opr>::ExeContext::ExeContext(                 \
            const FixedTensorLayouts& layouts, megdnn::Opr* megdnn_opr,        \
            const std::string& param_str, const cg::OperatorNodeBase* mgb_opr, \
            const CompNode& cn,                                                \
            const megdnn::param::ExecutionPolicy& execution_policy,            \
            bool allow_weight_preprocess);                                     \
    template typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy            \
    AlgoChooser<megdnn::Opr>::ExeContext::choose_by_heuristic(                 \
            ExecutionStrategy select_strategy) const;                          \
    template typename AlgoChooser<megdnn::Opr>::ImplAlgo                       \
    AlgoChooser<megdnn::Opr>::ExeContext::get_profile_result_from_cache(       \
            ExecutionStrategy select_strategy) const;                          \
    template std::vector<typename AlgoChooser<megdnn::Opr>::ImplAlgo>          \
    AlgoChooser<megdnn::Opr>::ExeContext::get_all_candidates() const;          \
    template size_t                                                            \
    AlgoChooser<megdnn::Opr>::ExeContext::get_workspace_size_bytes(            \
            const typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy&      \
                    policy) const;                                             \
    template void                                                              \
    AlgoChooser<megdnn::Opr>::ExeContext::construct_execution_policy(          \
            ExecutionStrategy select_strategy,                                 \
            typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy& policy,    \
            bool retrive_from_cache) const;                                    \
    template Maybe<AlgoChooserProfileCache::ResultEntry>                       \
    AlgoChooser<megdnn::Opr>::ExeContext::profile_single_algo(                 \
            const typename AlgoChooser<megdnn::Opr>::ImplExecutionPolicy&      \
                    policy,                                                    \
            double& timeout) const;

MGB_FOREACH_FASTRUN_OPR(INST)

#undef INST
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
