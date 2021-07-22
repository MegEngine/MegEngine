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

struct InternalGraph {
    ThinHashSet<VarNode*> m_internals;
    ThinHashSet<VarNode*> m_inputs;
    ThinHashSet<VarNode*> m_outputs;
};

class SubGraphExtractor {
public:
    using OprList = ThinHashSet<Typeinfo*>;
    SubGraphExtractor(OprList opr_list) : m_opr_list{opr_list} {};
    std::vector<InternalGraph> extract(
            const SymbolVarArray& endpoint_vars) const;

private:
    class Impl;
    OprList m_opr_list;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
