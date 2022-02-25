#pragma once

#include "megbrain/plugin/base.h"

namespace mgb {

/*!
 * \brief print warning if an operator does not call dispatch on cpu comp
 *      nodes
 *
 * This is intended to find potential bugs in megdnn.
 */
class CPUDispatchChecker final : public PluginBase {
    MGB_MUTEX m_cn2nr_task_mtx, m_failed_oprs_mtx_storage,
            *m_failed_oprs_mtx = &m_failed_oprs_mtx_storage;
    CompNode::UnorderedMap<size_t> m_cn2nr_task;
    std::unordered_set<cg::OperatorNodeBase*> m_failed_oprs_storage,
            *m_failed_oprs = &m_failed_oprs_storage;
    std::vector<std::unique_ptr<CPUDispatchChecker>> m_sub_graph_checkers;

    void record(CompNode cn);
    void check(CompNode cn, cg::OperatorNodeBase* opr);

public:
    MGE_WIN_DECLSPEC_FUC CPUDispatchChecker(cg::ComputingGraph* graph);

    //! get oprs that did not call cpu dispatch
    MGE_WIN_DECLSPEC_FUC auto&& failed_oprs() const { return *m_failed_oprs; }
};
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
