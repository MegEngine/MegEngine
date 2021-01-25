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
                bool reproducible = false) const;

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
        ImplAlgo get_profile_result_from_cache(bool require_reproducible) const;

        /**
         * \brief construct execution policy from cache or heuristic.
         *
         * \param require_reproducible select algo which is reproducible
         * \param policy execution policy
         * \param retrive_from_cache retrive algo from cache if set True, get
         *     from heuristic otherwise.
         */
        void construct_execution_policy(
                bool require_reproducible, ImplExecutionPolicy& policy,
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
    static void profile(ExeContext& ctx, bool require_reproducible);

    static ImplExecutionPolicy choose_by_profile(ExeContext& ctx,
                                                 bool require_reproducible,
                                                 bool enable_update = true);

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
    static std::vector<megdnn::Algorithm::SearchItem> flatten_search_space(
            const ExeContext& ctx);

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
