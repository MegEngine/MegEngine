/**
 * \file src/opr/impl/search_policy/algo_chooser.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/search_policy/algo_chooser.h"
#include "megbrain/opr/search_policy/profiler.h"

#include "../internal/invoke.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "./workspace_need_limit_getter.inl"

//! TODO: here has to be know some megdnn::opr when there is produced midout.h
//! fix it if there is another graceful way.
#include "megdnn/oprs.h"
#include "midout.h"
MIDOUT_DECL(megbrain_opr_algo_chooser)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_algo_chooser, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using mgb::opr::intl::WorkspaceLimitGetter;

#define APPLY(statement, ...)                                  \
    mgb::apply([&](const auto&... args) { return statement; }, \
               std::tuple_cat(__VA_ARGS__))

// timeout delta to be added with fastest known algorithm for new algos
constexpr double TIMEOUT_TOLERANCE = 2;

namespace mgb {
namespace opr {

template <typename Opr>
AlgoChooserProfileCache::Result AlgoChooser<Opr>::get_profile_result(
        ExeContext& ctx, bool enable_update) {
    AlgoChooserProfileCache& cache = ctx.mgb_opr()->profile_cache();

    ConvTensorLayouts origin_layouts = ctx.layouts();
    typename Opr::Param origin_param = ctx.mgb_opr()->param();
    get_origin_param_and_layouts(ctx, origin_layouts, origin_param);
    AlgoChooserProfileCache::Key cache_key{origin_layouts.data(),
                                           origin_layouts.size(), &origin_param,
                                           sizeof(origin_param)};
    {
        auto&& rst = cache.get(cache_key);
        if (rst.valid())
            return rst.val();
    }

    AlgoChooserProfileCache::Result prof_rst;
    if (!enable_update)
        return prof_rst;

    std::string str_on_inp_shape = ssprintf(
            "on input layouts (%s, %s)", ctx.layouts()[0].to_string().c_str(),
            ctx.layouts()[1].to_string().c_str());
    double cur_timeout = 0;
    RealTimer timer;
    for (auto algo : ctx.get_all_candidates_with_workspace_limit()) {
        Maybe<AlgoChooserProfileCache::ResultEntry> cur_rst;
        std::string msg = ssprintf("profiling %s algorithm %s %s",
                                   ctx.mgb_opr()->dyn_typeinfo()->name,
                                   algo.name.c_str(), str_on_inp_shape.c_str());
        timer.reset();
        MGB_TRY { cur_rst = ctx.profile_single_algo(algo, cur_timeout); }
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
    mgb_assert(!prof_rst.empty(), "no usable convolution algorithm %s",
               str_on_inp_shape.c_str());

    cache.put(cache_key, prof_rst);
    return prof_rst;
}

template <>
void AlgoChooser<megdnn::ConvBias>::get_origin_param_and_layouts(
        const ExeContext& ctx, ConvTensorLayouts& layouts,
        megdnn::ConvBias::Param& param) {
    auto format = static_cast<megdnn::param::ConvBias::Format>(
            ctx.megdnn_opr()->param().format);
    size_t output_block_size = ctx.megdnn_opr()->param().output_block_size;
    megdnn::ConvBias::deduce_winograd_origin_layout_and_param(
            format, output_block_size, ctx.layouts()[0], ctx.layouts()[1],
            layouts[1], param);
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplAlgo AlgoChooser<Opr>::choose_by_profile(
        ExeContext& ctx, bool require_reproducible, bool enable_update) {
    MIDOUT_B(Opr, midout_iv(MGB_HASH_STR("AlgoChooser::choose_by_profile")))
    auto opr = ctx.mgb_opr();
    if (opr->owner_graph()->options().no_profiling_on_shape_change) {
        auto algo = ctx.megdnn_opr()->execution_policy().algo;
        if (algo.valid())
            return algo;
    }

    std::unordered_map<std::string, ImplAlgo> algo_map;
    for (auto i : ctx.get_all_candidates()) {
        auto ins = algo_map.emplace(i.name.c_str(), i);
        mgb_assert(ins.second, "duplicated algo name: %s", i.name.c_str());
    }

    auto&& prof = get_profile_result(ctx, enable_update);
    if (prof.empty())
        return {};
    for (auto&& i : prof) {
        if ((!require_reproducible || i.reproducible)) {
            auto iter = algo_map.find(i.algo);
            mgb_assert(iter != algo_map.end(),
                       "algorithm %s exists in "
                       "profiling result but not in algo_map; please "
                       "report this "
                       "bug; opr: %s{%s}, shapes: %s %s %s",
                       ctx.mgb_opr()->cname(),
                       ctx.mgb_opr()->dyn_typeinfo()->name,
                       ctx.layouts()[0].TensorShape::to_string().c_str(),
                       ctx.layouts()[1].TensorShape::to_string().c_str(),
                       ctx.layouts()[2].TensorShape::to_string().c_str(),
                       i.algo.c_str());
            return iter->second;
        }
    }

    mgb_log_error(
            "Workspace requirement (%zu) could not be satisfied. Abort now "
            "to "
            "avoid further problems",
            WorkspaceLimitGetter::get_workspace_limit(
                    opr->owner_graph(), opr->comp_node(),
                    opr->execution_policy().workspace_limit));
    mgb_trap();
    MIDOUT_E
}

template <typename Opr>
size_t AlgoChooser<Opr>::setup_algo(const ConvTensorLayouts& layouts,
                                    Opr* megdnn_opr, const MGBOpr* mgb_opr,
                                    bool allow_weight_preprocess) {
    if (WorkspaceLimitGetter::is_prealloc_run(mgb_opr->owner_graph())) {
        return 0;
    }

    ImplAlgo algo = {};
    ExeContext ctx(layouts, megdnn_opr, mgb_opr, allow_weight_preprocess);

    if (auto algo_choose_hook = mgb_opr->algo_chooser()) {
        algo = algo_choose_hook(mgb_opr);
    }
    if (!algo.valid()) {
        algo = get_algo(ctx);
    }
    size_t workspace = ctx.get_workspace_size_bytes(algo);
    mgb_log_debug(
            "%s: tensor layouts(%s %s, %s %s) -> (%s %s): algo=%s "
            "workspace=%.2fMiB reproducible=%d",
            mgb_opr->dyn_typeinfo()->name, layouts[0].to_string().c_str(),
            layouts[0].dtype.name(), layouts[1].to_string().c_str(),
            layouts[1].dtype.name(),
            layouts[layouts.size() - 1].to_string().c_str(),
            layouts[layouts.size() - 1].dtype.name(), algo.name.c_str(),
            workspace / (1024 * 1024.0), algo.is_reproducible);
    megdnn_opr->execution_policy() = {algo};
    return workspace;
}

template <typename Opr>
typename AlgoChooser<Opr>::ImplAlgo AlgoChooser<Opr>::get_algo(
        ExeContext& ctx) {
    using S = mixin::Convolution::ExecutionPolicy::Strategy;
    MGB_MARK_USED_VAR(TIMEOUT_TOLERANCE);
    switch (ctx.mgb_opr()->execution_policy().strategy) {
        case S::HEURISTIC:
            return ctx.choose_by_heuristic();
        case S::HEURISTIC_REPRODUCIBLE:
            return ctx.choose_by_heuristic(true);
        case S::PROFILE_HEURISTIC: {
            ImplAlgo algo = choose_by_profile(ctx, false, false);
            if (!algo.valid())
                algo = ctx.choose_by_heuristic();
            return algo;
        }
#if MGB_ENABLE_FASTRUN
        case S::PROFILE:
            return choose_by_profile(ctx, false);
        case S::PROFILE_REPRODUCIBLE:
            return choose_by_profile(ctx, true);
#endif
        default:
            mgb_throw(GraphError, "bad convolution ExecutionPolicy strategy");
    }
}

#define INST(Opr)                                                            \
    template AlgoChooser<megdnn::Opr>::ImplAlgo                              \
    AlgoChooser<megdnn::Opr>::get_algo(ExeContext& ctx);                     \
    template AlgoChooserProfileCache::Result                                 \
    AlgoChooser<megdnn::Opr>::get_profile_result(ExeContext& ctx,            \
                                                 bool enable_update);        \
    template AlgoChooser<megdnn::Opr>::ImplAlgo                              \
    AlgoChooser<megdnn::Opr>::choose_by_profile(                             \
            ExeContext& ctx, bool require_reproducible, bool enable_update); \
    template size_t AlgoChooser<megdnn::Opr>::setup_algo(                    \
            const ConvTensorLayouts& layouts, megdnn::Opr* megdnn_opr,       \
            const MGBOpr* mgb_opr, bool allow_weight_preprocess);

MGB_FOREACH_FASTRUN_OPR(INST)

#undef INST

//////////////////////////////// ExeContext /////////////////////////////

template <typename Opr>
typename AlgoChooser<Opr>::ImplAlgo
AlgoChooser<Opr>::ExeContext::choose_by_heuristic(bool reproducible) const {
    auto opr = m_mgb_opr;
    auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
            opr->owner_graph(), opr->comp_node(),
            opr->execution_policy().workspace_limit);
    return APPLY(m_megdnn_opr->get_algorithm_info_heuristic(
                         args..., workspace_limit, reproducible),
                 m_layouts);
}

template <typename Opr>
std::vector<typename AlgoChooser<Opr>::ImplAlgo>
AlgoChooser<Opr>::ExeContext::get_all_candidates() const {
    auto heu = choose_by_heuristic();
    auto&& ret =
            APPLY(m_megdnn_opr->get_all_algorithms_info(args...), m_layouts);
    bool found = false;
    for (size_t i = 0; i < ret.size(); ++i) {
        if (ret[i] == heu) {
            found = true;
            std::swap(ret[i], ret[0]);
            break;
        }
    }
    mgb_assert(found,
               "algo %s got by heuristic not found in "
               "candidate list",
               heu.name.c_str());
    return std::move(ret);
}

template <typename Opr>
std::vector<typename AlgoChooser<Opr>::ImplAlgo>
AlgoChooser<Opr>::ExeContext::get_all_candidates_with_workspace_limit() const {
    auto&& all_algos = get_all_candidates();
    auto opr = m_mgb_opr;
    auto workspace_limit = WorkspaceLimitGetter::get_workspace_limit(
            opr->owner_graph(), opr->comp_node(),
            opr->execution_policy().workspace_limit);
    std::vector<ImplAlgo> ret;
    for (auto&& algo : all_algos) {
        if (get_workspace_size_bytes(algo) <= workspace_limit) {
            ret.push_back(algo);
        }
    }
    return ret;
}

template <typename Opr>
size_t AlgoChooser<Opr>::ExeContext::get_workspace_size_bytes(
        ImplAlgo algo) const {
    m_megdnn_opr->execution_policy() = {algo};
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
AlgoChooser<Opr>::ExeContext::profile_single_algo(ImplAlgo algo,
                                                  double& timeout) const {
    typename TimedProfiler<Opr>::Param param;
    auto name = algo.name.c_str();
    // force check copy size <= dest len-1 from gcc8 for safe
    auto len = sizeof(param.algo_name);
    strncpy(param.algo_name, name, len - 1);
    param.algo_name[len - 1] = '\0';
    mgb_assert(!param.algo_name[sizeof(param.algo_name) - 2],
               "algo name too long: %s; len=%zu", name, strlen(name));
    param.workspace = get_workspace_size_bytes(algo);
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
    param.comp_node_loc = m_mgb_opr->output(0)->comp_node().locator();
    mgb_assert(param.shapes.size() == m_layouts.size());
    for (size_t i = 0; i < param.shapes.size(); ++i)
        param.shapes[i] = m_layouts[i];
    param.opr_param = m_megdnn_opr->param();
    param.allow_weight_preprocess = m_allow_weight_preprocess;

    auto rst = TimedProfiler<Opr>::profile(param, timeout);
    // MIOpen conv profiles all available algos when a specfic shape is
    // provided for the first time, which probably adds to the result time.
    // Therefore, a second profile execution is needed.
    if (strncmp(name, "MIOpen", 6) == 0)
        rst = TimedProfiler<Opr>::profile(param, timeout);
    if (!rst.valid())
        return None;
    return AlgoChooserProfileCache::ResultEntry{
            algo.name.c_str(), algo.is_reproducible, rst.val().time,
            param.workspace};
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
    template typename AlgoChooser<megdnn::Opr>::ImplAlgo                       \
    AlgoChooser<megdnn::Opr>::ExeContext::choose_by_heuristic(                 \
            bool reproducible) const;                                          \
    template std::vector<typename AlgoChooser<megdnn::Opr>::ImplAlgo>          \
    AlgoChooser<megdnn::Opr>::ExeContext::get_all_candidates() const;          \
    template std::vector<typename AlgoChooser<megdnn::Opr>::ImplAlgo>          \
    AlgoChooser<megdnn::Opr>::ExeContext::                                     \
            get_all_candidates_with_workspace_limit() const;                   \
    template size_t                                                            \
    AlgoChooser<megdnn::Opr>::ExeContext::get_workspace_size_bytes(            \
            typename AlgoChooser<megdnn::Opr>::ImplAlgo algo) const;           \
    template Maybe<AlgoChooserProfileCache::ResultEntry>                       \
    AlgoChooser<megdnn::Opr>::ExeContext::profile_single_algo(                 \
            typename AlgoChooser<megdnn::Opr>::ImplAlgo algo, double& timeout) \
            const;                                                             \

MGB_FOREACH_FASTRUN_OPR(INST)

#undef INST
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
