/**
 * \file src/opr/include/megbrain/opr/search_policy/algo_chooser.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <memory>
#include "megbrain/graph/cg.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/opr/search_policy/profiler.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/blas.h"
#include "megdnn/oprs/base.h"

template <class MegDNNOpr>
struct MegDNNOpr2MGBOpr;

#define cb(_Opr)                            \
    template <>                             \
    struct MegDNNOpr2MGBOpr<megdnn::_Opr> { \
        using MGBOpr = mgb::opr::_Opr;      \
    };

MGB_FOREACH_FASTRUN_OPR(cb)

#undef cb

namespace mgb {

//! define logical operation of megdnn::param::ExecutionPolicy::Strategy::Enum
//! and megdnn::detail::AlgoAttribute enum
using ExecutionStrategy = megdnn::param::ExecutionPolicy::Strategy;

using AlgoAttribute = megdnn::AlgoAttribute;

namespace opr {

/* =================== AlgoChooser =================== */
/*!
 * \brief choose algorithm according to ExecutionPolicy
 *
 * This class only provides static methods, and the entry point is
 * AlgoChooser::setup_algo. When profiling is needed, it would first try to
 * retrive profiling stats from cache, and run TimedProfiler when necessary
 *
 * \tparam Opr megdnn operator impl
 */
template <typename Opr>
class AlgoChooser {
    static constexpr int arity_in = OprArityTrait<Opr>::arity_in;
    static constexpr int arity_out = OprArityTrait<Opr>::arity_out;
    static constexpr int arity = OprArityTrait<Opr>::arity;

    using ImplAlgo = typename Opr::AlgorithmInfo;
    using ImplExecutionPolicy = megdnn::ExecutionPolicy;
    using MGBOpr = typename MegDNNOpr2MGBOpr<Opr>::MGBOpr;

public:
    using FixedTensorLayouts = std::array<TensorLayout, arity>;
    class ExeContext {
        FixedTensorLayouts m_layouts;
        Opr* m_megdnn_opr;
        std::string m_param;
        const cg::OperatorNodeBase* m_base_mgb_opr;
        CompNode m_cn;
        megdnn::param::ExecutionPolicy m_execution_policy;
        bool m_allow_weight_preprocess;

    public:
        ExeContext(const FixedTensorLayouts& layouts, Opr* megdnn_opr,
                   const std::string& param_str,
                   const cg::OperatorNodeBase* mgb_opr, const CompNode& cn,
                   const megdnn::param::ExecutionPolicy& execution_policy,
                   bool allow_weight_preprocess);

        Opr* megdnn_opr() const { return m_megdnn_opr; }

        const TensorLayout& inp_layout(size_t idx) const {
            return m_layouts[idx];
        }

        cg::ComputingGraph* owner_graph() const {
            return m_base_mgb_opr->owner_graph();
        }
        const cg::OperatorNodeBase* mgb_opr() const { return m_base_mgb_opr; }
        const megdnn::param::ExecutionPolicy& execution_policy() const {
            return m_execution_policy;
        }
        CompNode comp_node() const { return m_cn; }
        const std::string& param() const { return m_param; }

        bool allow_weight_preprocess() const {
            return m_allow_weight_preprocess;
        }

        megdnn::Algorithm* get_algorithm_from_desc(
                const megdnn::Algorithm::Info::Desc& desc) const {
            return m_megdnn_opr->get_algorithm_from_desc(desc);
        }

        const FixedTensorLayouts& layouts() const { return m_layouts; }

        ImplExecutionPolicy choose_by_heuristic(
                ExecutionStrategy selected_strategy) const;

        //! get all candidate algos, and the one choose_by_heuristic() is
        //! put first
        std::vector<ImplAlgo> get_all_candidates() const;

        //! get workspace size required for specific execution policy
        size_t get_workspace_size_bytes(
                const ImplExecutionPolicy& policy) const;

        /*!
         * \brief profile a single algorithm
         *
         * This is actually a wrapper that constructs param and call
         * TimedProfiler<Opr>::profile for the actual profiling
         *
         * \param[in,out] timeout set the timeout, and return the actual
         *      timeout used during profiling
         */
        Maybe<AlgoChooserProfileCache::ResultEntry> profile_single_algo(
                const ImplExecutionPolicy& policy, double& timeout) const;

        //! get all profile algorithm from cache, return invalid if not exists
        ImplAlgo get_profile_result_from_cache(
                ExecutionStrategy selected_strategy) const;

        /**
         * \brief construct execution policy from cache or heuristic.
         *
         * \param selected_strategy select algo which matched this strategy
         * \param policy execution policy
         * \param retrive_from_cache retrive algo from cache if set True, get
         *     from heuristic otherwise.
         */
        void construct_execution_policy(ExecutionStrategy selected_strategy,
                                        ImplExecutionPolicy& policy,
                                        bool retrive_from_cache = true) const;

    private:
        Maybe<PreprocessFilter<Opr>> construct_fake_preprocess_filter() const;
    };

    template<typename U>
    friend class AlgoChooser;

private:
    //! entrance for getting algorithm according to execution strategy
    static ImplExecutionPolicy get_policy(ExeContext& ctx);


    //! profile and save to cache
    static void profile(ExeContext& ctx, ExecutionStrategy selected_strategy);

    static ImplExecutionPolicy choose_by_profile(
            ExeContext& ctx, ExecutionStrategy selected_strategy,
            bool enable_update = true);

public:
    /*!
     * \brief setup algorithm and return workspace size
     */
    static size_t setup_algo(const FixedTensorLayouts& layouts, Opr* megdnn_opr,
                             const MGBOpr* mgb_opr,
                             bool allow_weight_preprocess = false);
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
