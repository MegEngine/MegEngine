/**
 * \file src/gopt/impl/subgraph_extractor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/subgraph_extractor.h"

using namespace mgb;
using namespace cg;
using namespace gopt;

/* ================== SubGraphExtractor =================*/
std::vector<InternalGraph> SubGraphExtractor::extract(
        const SymbolVarArray& endpoint_vars) const {
    ThinHashMap<OperatorNodeBase*, std::pair<OperatorNodeBase*, int>> parent;
    thin_function<OperatorNodeBase*(OperatorNodeBase*)> union_find;
    auto union_find = [&parent, &union_find](OperatorNodeBase* o) {
        if (parent[o].first == o)
            return o;
        else {
            auto p = union_find(parent[o].first);
            parent[o].first = p;
            return p;
        }
    };
    auto union_merge = [&parent, &union_find](OperatorNodeBase* x,
                                              OperatorNodeBase* y) {
        auto root_x = union_find(x), root_y = union_find(y);
        if (root_x != root_y) {
            OperatorNodeBase *large, small;
            if (parent[root_x].second < parent[root_y].second) {
                small = root_x, large = root_y;
            } else {
                small = root_y, large = root_x;
            }
            parent[small].first = large;
            if (parent[large].second == parent[small].second) {
                parend[large].second += 1;
            }
        }
    };

    std::vector<OperatorNodeBase*> topo;
    auto cb = [&topo](OperatorNodeBase* opr) {
        topo.push_back(opr);
        if (opr_list.count(opr->dyn_typeinfo()) == 0)
            return;
        auto find = parent.find(opr);
        if (find == parent.end()) {
            auto insert =
                    parent.insert(std::make_pair(opr, std::make_pair(opr, 0)));
            find = insert.first;
        }
        for (auto&& i : opr->input()) {
            auto&& o = i->owner_opr();
            if (opr_list.count(o->dyn_typeinfo()) == 0)
                continue;
            union_merge(opr, o);
        }
    };
    cg::DepOprIter iter{cb};
    for (const auto& v : endpoint_vars)
        iter.add(v.node()->owner_opr());

    std::vector<InternalGraph> partitions;
    ThinHashMap<OperatorNodeBase*, InternalGraph*> roots;
    for (const auto& opr : reverse_adaptor(topo)) {
        auto root = union_find(opr);
        auto find = roots.find(root);
        InternalGraph* internal_graph = nullptr;
        if (find == roots.end()) {
            partitions.emplace_back(InternalGraph{});
            auto insert =
                    roots.insert(std::make_pair(root, &partitions.back()));
            internal_graph = insert.first->second;
            internal_graph->m_outputs.insert(opr->output(0));
        } else {
            internal_graph = find->second;
            auto erase = internal_graph->m_inputs.erase(opr->output(0));
            if (erase > 0) {
                internal_graph->m_internals.insert(opr->output(0));
            } else {
                internal_graph->m_outputs.insert(opr->output(0));
            }
        }
        for (const auto& i : opr->input())
            internal_graph->m_inputs.insert(i);
    }
    return partitions;
}

/* ============= SubGraphExtractor =================*/

// vim: syntax=cpp.doxygen
