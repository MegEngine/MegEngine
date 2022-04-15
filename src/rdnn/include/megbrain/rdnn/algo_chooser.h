#pragma once

#include <memory>
#include "megbrain/opr/param_defs.h"
#include "megbrain/rdnn/profiler.h"
#include "megbrain/utils/persistent_cache.h"
#include "megdnn/oprs/base.h"

#define CACHE_KEY_VERSION "v5"
namespace mgb {
namespace rdnn {

//! define logical operation of megdnn::param::ExecutionPolicy::Strategy::Enum
//! and megdnn::detail::AlgoAttribute enum
using ExecutionStrategy = megdnn::param::ExecutionPolicy::Strategy;

using AlgoAttribute = megdnn::AlgoAttribute;

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
struct AlgoChooserDesc {
    uint32_t shared_batch_size = 0;
    bool binary_equal_between_batch = false;
    bool no_profiling_on_shape_change = false;
    using WorkspaceLimitGetter = std::function<size_t(CompNode, size_t)>;
    WorkspaceLimitGetter get_workspace_limit;
};

template <typename Opr>
class AlgoChooser {
    static constexpr int arity_in = OprArityTrait<Opr>::arity_in;
    static constexpr int arity_out = OprArityTrait<Opr>::arity_out;
    static constexpr int arity = OprArityTrait<Opr>::arity;

    using ImplAlgo = typename Opr::AlgorithmInfo;
    using ImplAlgoDesc = typename Opr::AlgorithmInfo::Desc;

protected:
    using ImplExecutionPolicy = megdnn::ExecutionPolicy;

public:
    using FixedTensorLayouts = std::array<TensorLayout, arity>;

    class AlgoChooserHelper {
        //! fastrun layouts
        FixedTensorLayouts m_fastrun_layouts;
        //! layouts used when get and set cache item
        FixedTensorLayouts m_incache_layouts;
        Opr* m_dnn_opr;
        std::string m_param;
        CompNode m_cn;
        megdnn::param::ExecutionPolicy m_execution_policy;
        bool m_allow_weight_preprocess;
        const AlgoChooserDesc& m_desc;

    public:
        MGE_WIN_DECLSPEC_FUC AlgoChooserHelper(
                const FixedTensorLayouts& layouts, Opr* megdnn_opr,
                const std::string& param_str, const CompNode& cn,
                const megdnn::param::ExecutionPolicy& execution_policy,
                bool allow_weight_preprocess, const AlgoChooserDesc& desc);

        Opr* megdnn_opr() const { return m_dnn_opr; }

        const TensorLayout& inp_layout(size_t idx) const {
            return m_fastrun_layouts[idx];
        }

        const megdnn::param::ExecutionPolicy& execution_policy() const {
            return m_execution_policy;
        }
        CompNode comp_node() const { return m_cn; }
        const std::string& param() const { return m_param; }

        bool allow_weight_preprocess() const { return m_allow_weight_preprocess; }

        megdnn::Algorithm* get_algorithm_from_desc(
                const megdnn::Algorithm::Info::Desc& desc) const {
            return m_dnn_opr->get_algorithm_from_desc(desc);
        }

        const FixedTensorLayouts& fastrun_layouts() const { return m_fastrun_layouts; }

        const FixedTensorLayouts& incache_layouts() const { return m_incache_layouts; }

        const AlgoChooserDesc& desc() const { return m_desc; }

        //! construct algo chain by heuristic
        ImplExecutionPolicy choose_by_heuristic(
                const ExecutionStrategy& selected_strategy) const;

        //! construct algo chain by profiling
        ImplExecutionPolicy choose_by_profile(
                const ExecutionStrategy& selected_strategy, bool enable_update) const;

        //! get all profile algorithm from cache, return invalid if not exists
        std::pair<ImplAlgoDesc, Maybe<AlgoChooserProfileCache::Result>>
        get_profile_result_from_cache(const ExecutionStrategy& selected_strategy) const;

        /**
         * \brief construct execution policy from cache or heuristic.
         *
         * \param selected_strategy select algo which matched this strategy
         * \param[in,out] policy execution policy
         * \param retrive_from_cache retrive algo from cache if set True, get
         *     from heuristic otherwise.
         * \param allow_log no warning log print if set True, print warning info
         * otherwise.
         */
        void construct_execution_policy(
                const ExecutionStrategy& selected_strategy, ImplExecutionPolicy& policy,
                bool retrive_from_cache = true, bool allow_log = true) const;

        //! get workspace size required for specific execution policy
        MGE_WIN_DECLSPEC_FUC size_t get_workspace_size_bytes(
                const ImplExecutionPolicy& policy,
                const FixedTensorLayouts& layouts = {}) const;

        //! get all candidate algos, and the one choose_by_heuristic() is
        //! put first
        std::vector<ImplAlgo> get_all_candidates() const;

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

        //! profile and save to cache
        void profile(const ExecutionStrategy& selected_strategy) const;

        /**
         * \brief extract algo attribute from execution strategy and graph
         * option.
         *
         * \param strategy select algo which matched this strategy
         * \return pair<positive_attr, negative_attr>
         */
        std::pair<AlgoAttribute, AlgoAttribute> extract_algo_attribute(
                const ExecutionStrategy& strategy) const;

    private:
        Maybe<PreprocessFilter<Opr>> construct_fake_preprocess_filter(
                const FixedTensorLayouts& layouts = {}) const;
    };

    template <typename U>
    friend class AlgoChooser;

    //! entrance for getting algorithm according to execution strategy
    MGE_WIN_DECLSPEC_FUC static ImplExecutionPolicy get_policy(
            const AlgoChooserHelper& helper);

    //! format given layouts to string
    static std::string format_fixlayouts(const FixedTensorLayouts& layout);
};

}  // namespace rdnn
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
