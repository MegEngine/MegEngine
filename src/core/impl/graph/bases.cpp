#include "megbrain/graph/bases.h"
#include "./cg_impl.h"

using namespace mgb::cg;

MGB_TYPEINFO_OBJ_IMPL(OutputVarsUserData);

GraphNodeBase::GraphNodeBase(ComputingGraph* owner_graph) : m_owner_graph{owner_graph} {
    mgb_assert(owner_graph, "owner graph not given");
    m_id = owner_graph->next_node_id();
}

AsyncExecutable::~AsyncExecutable() noexcept = default;

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
