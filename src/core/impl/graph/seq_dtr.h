#pragma once

#include "./memory_optimizer.h"
#include "./seq_modifier_base.h"
#include "megbrain/graph/cg.h"

#if MGB_ENABLE_DTR

namespace mgb {
namespace cg {

class SeqModifierForDTR : public SeqModifierBase {
    //! Config options
    using Config = mgb::cg::ComputingGraph::Options::DTRConfig;
    Config* m_config;

    class ModifyActionPlanner;

public:
    SeqModifierForDTR(ComputingGraphImpl* owner, Config* config_g);

    void modify_endpoint_vars(VarNodeArray& endpoints);

    void apply_action(SeqModifyAction& action, const OprNodeArray& oprseq);
};

}  // namespace cg
}  // namespace mgb

#endif  //  MGB_ENABLE_DTR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
