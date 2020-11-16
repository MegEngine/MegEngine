/**
 * \file src/opr/include/megbrain/opr/search_policy/algo_chooser.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain/opr/search_policy/profiler.h"

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
    using MGBOpr = typename MegDNNOpr2MGBOpr<Opr>::MGBOpr;
    using ConvTensorLayouts = std::array<TensorLayout, arity>;

    class ExeContext {
        const ConvTensorLayouts& m_layouts;
        Opr* m_megdnn_opr;
        const MGBOpr* m_mgb_opr;
        bool m_allow_weight_preprocess;

    public:
        ExeContext(const ConvTensorLayouts& layouts, Opr* megdnn_opr,
                   const MGBOpr* mgb_opr, bool allow_weight_preprocess)
                : m_layouts{layouts},
                  m_megdnn_opr{megdnn_opr},
                  m_mgb_opr{mgb_opr},
                  m_allow_weight_preprocess{allow_weight_preprocess} {
            mgb_assert(m_layouts.size() == layouts.size());
            static_assert(
                    std::tuple_size<ConvTensorLayouts>::value == 3 ||
                            std::tuple_size<ConvTensorLayouts>::value == 5 ||
                            std::tuple_size<ConvTensorLayouts>::value == 8,
                    "Convolution AlgoChooser assumes arity = 3 , 5 or 8 (for "
                    "deformable conv)");
        }

        Opr* megdnn_opr() const { return m_megdnn_opr; }

        const MGBOpr* mgb_opr() const { return m_mgb_opr; }

        const TensorLayout& inp_layout(size_t idx) const {
            return m_layouts[idx];
        }

        const ConvTensorLayouts& layouts() const { return m_layouts; }

        ImplAlgo choose_by_heuristic(bool reproducible = false) const;

        //! get all candidate algos, and the one choose_by_heuristic() is
        //! put first
        std::vector<ImplAlgo> get_all_candidates() const;

        //! get candidate algos with workspace limit.
        std::vector<ImplAlgo> get_all_candidates_with_workspace_limit() const;

        //! get workspace size required for specific algo
        size_t get_workspace_size_bytes(ImplAlgo algo) const;

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
                ImplAlgo algo, double& timeout) const;

    private:
        Maybe<PreprocessFilter<Opr>> construct_fake_preprocess_filter() const;
    };

    //! entrance for getting algorithm according to execution strategy
    static ImplAlgo get_algo(ExeContext& ctx);

    static void get_origin_param_and_layouts(const ExeContext&,
                                             ConvTensorLayouts&,
                                             typename Opr::Param&) {}

    //! get all profile result, either by retrieving cache or profiling
    static AlgoChooserProfileCache::Result get_profile_result(
            ExeContext& ctx, bool enable_update);

    static ImplAlgo choose_by_profile(ExeContext& ctx,
                                      bool require_reproducible,
                                      bool enable_update = true);

public:
    /*!
     * \brief setup algorithm and return workspace size
     */
    static size_t setup_algo(const ConvTensorLayouts& layouts, Opr* megdnn_opr,
                             const MGBOpr* mgb_opr,
                             bool allow_weight_preprocess = false);
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
