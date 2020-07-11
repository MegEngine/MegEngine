/**
 * \file src/core/impl/graph/bases.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/bases.h"
#include "./cg_impl.h"

using namespace mgb::cg;

GraphNodeBase::GraphNodeBase(ComputingGraph *owner_graph):
    m_owner_graph{owner_graph}
{
    mgb_assert(owner_graph, "owner graph not given");
    m_id = owner_graph->next_node_id();
}

AsyncExecutable::~AsyncExecutable() noexcept = default;

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
