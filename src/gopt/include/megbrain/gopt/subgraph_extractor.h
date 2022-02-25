#pragma once
#include "megbrain/graph.h"

namespace mgb {
namespace gopt {

class GraphPartition {
public:
    using VarNodeSet = ThinHashSet<VarNode*>;
    using OperatorNodeSet = ThinHashSet<cg::OperatorNodeBase*>;
    using OperatorNodeList = std::vector<cg::OperatorNodeBase*>;

    class InputPlaceholder;

    GraphPartition() = default;

#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> to_json() const;
#endif

    const OperatorNodeSet& opr_set() const { return m_opr_set; }
    const VarNodeSet& input() const { return m_inputs; }
    const VarNodeSet& output() const { return m_outputs; }
    const OperatorNodeList& all_oprs() const { return m_oprs; }
    OperatorNodeSet& opr_set() { return m_opr_set; }
    OperatorNodeList& all_oprs() { return m_oprs; }
    VarNodeSet& input() { return m_inputs; }
    VarNodeSet& output() { return m_outputs; }

private:
    std::pair<VarNodeArray, VarNodeArray> replace_graph_by_placeholder() const;
    OperatorNodeSet m_opr_set;
    OperatorNodeList m_oprs;
    VarNodeSet m_inputs;
    VarNodeSet m_outputs;
};

class SubGraphExtractor {
public:
    using OprList = ThinHashSet<Typeinfo*>;
    SubGraphExtractor(const OprList& opr_list) : m_opr_list{opr_list} {};
    std::vector<GraphPartition> extract(const SymbolVarArray& endpoint_vars) const;

private:
    class Impl;
    const OprList& m_opr_list;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
