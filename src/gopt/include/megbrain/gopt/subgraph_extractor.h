/**
 * \file src/gopt/include/megbrain/gopt/subgraph_extractor.h
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
#include "megbrain/graph.h"

namespace mgb {
namespace gopt {

class GraphPartition {
public:
    using VarNodeSet = ThinHashSet<VarNode*>;
    using OperatorNodeSet = ThinHashSet<cg::OperatorNodeBase*>;

    class InputPlaceholder;

    GraphPartition() = default;

#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> to_json() const;
#endif

    const OperatorNodeSet& opr_set() const { return m_opr_set; }
    const VarNodeSet& input() const { return m_inputs; }
    const VarNodeSet& output() const { return m_outputs; }
    OperatorNodeSet& opr_set() { return m_opr_set; }
    VarNodeSet& input() { return m_inputs; }
    VarNodeSet& output() { return m_outputs; }

private:
    OperatorNodeSet m_opr_set;
    VarNodeSet m_inputs;
    VarNodeSet m_outputs;
    std::pair<VarNodeArray, VarNodeArray> replace_graph_by_placeholder() const;
};

class SubGraphExtractor {
public:
    using OprList = ThinHashSet<Typeinfo*>;
    SubGraphExtractor(const OprList& opr_list) : m_opr_list{opr_list} {};
    std::vector<GraphPartition> extract(
            const SymbolVarArray& endpoint_vars) const;

private:
    class Impl;
    const OprList& m_opr_list;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
