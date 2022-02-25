#pragma once

#include "./memory_optimizer.h"
#include "./seq_modifier_base.h"
#include "megbrain/graph/cg.h"
#include "megbrain/utils/async_worker.h"

#if MGB_ENABLE_SUBLINEAR
namespace mgb {
namespace cg {

/*!
 * \brief modifying computing sequence, with basically the same idea of Training
 *      Deep Nets with Sublinear Memory Cost
 */
class SeqModifierForSublinearMemory : public SeqModifierBase {
    //! Config options
    using Config = mgb::cg::ComputingGraph::Options::SublinearMemConfig;
    Config* m_config;

public:
    SeqModifierForSublinearMemory(ComputingGraphImpl* owner, Config* config_g);

    //! replace endpoint vars by the ones that require more computing
    void modify_endpoint_vars(VarNodeArray& endpoints);

    //! check whether actual opr_seq is what we expect; throw InternalError
    void sanity_check(const OprNodeArray& opr_seq);

    const CompNode::UnorderedMap<size_t>& prev_min_bottleneck();

private:
    using SplitPointSet = std::shared_ptr<std::vector<size_t>>;

    //! get modifications to be taken under some specific constraints
    class ModifyActionPlanner;

    //! search best modify action for opr seq on a single comp node
    class ActionSearcherSingleCN;

    struct InternalDeleter {
        void operator()(ActionSearcherSingleCN*) const;
        void operator()(ModifyActionPlanner*) const;
    };

    struct OprReplaceInfo {
        OperatorNodeBase *recomp = nullptr,  //!< recomp operator from replaced input
                *dup = nullptr;              //!< duplicated operator due to discarding
    };

    //! map from original operator to its replace info; used for sanity check
    ThinHashMap<OperatorNodeBase*, OprReplaceInfo> m_opr2replace_info;

    //! map from thread ID to corresponding ModifyActionPlanner as a worker
    std::unordered_map<
            std::thread::id, std::unique_ptr<ModifyActionPlanner, InternalDeleter>>
            m_thread2planner;

    //! thread pool to run ModifyActionPlanner
    FutureThreadPool<void> m_planner_thread_pool;

    CompNode::UnorderedMap<size_t> m_prev_min_bottleneck;

    //! restore computing sequence and modify operator priority
    void reset_opr_seq(const OprNodeArray& oprseq);

    //! search for best action based on *cn2oprseq*
    SeqModifyAction search_action(
            const CompNode::UnorderedMap<OprNodeArray>* cn2oprseq);

    //! apply action and store result to m_var_map
    void apply_action(SeqModifyAction& action, const OprNodeArray& oprseq);

    template <typename... Args>
    static SplitPointSet make_split_point_set(Args&&... args) {
        return std::make_shared<SplitPointSet::element_type>(
                std::forward<Args>(args)...);
    }
};

}  // namespace cg
}  // namespace mgb

#endif  //  MGB_ENABLE_SUBLINEAR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
