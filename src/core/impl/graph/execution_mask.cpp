/**
 * \file src/core/impl/graph/execution_mask.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl.h"

#include "megbrain/common.h"
#include "megbrain/graph/execution_mask.h"

using namespace mgb;
using namespace cg;

#if MGB_ENABLE_COND_EXEC

MGB_TYPEINFO_OBJ_IMPL(ExecutionMask);

std::atomic_size_t ExecutionMask::sm_tot_id{0};
std::atomic_size_t ExecutionMask::sm_alive_inst{0};

class ExecutionMask::RefHolder final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    SmallVector<std::shared_ptr<ExecutionMask>> m_refs;

public:
    static RefHolder& get(ComputingGraph* graph) {
        return *graph->options().user_data.get_user_data_or_create<RefHolder>();
    }

    void add(std::shared_ptr<ExecutionMask> mask) {
        m_refs.emplace_back(std::move(mask));
    }
};
MGB_TYPEINFO_OBJ_IMPL(ExecutionMask::RefHolder);

ExecutionMask::ExecutionMask(VarNode* owner)
        : m_id{sm_tot_id.fetch_add(1, std::memory_order_relaxed) + 1},
          m_owner{owner} {
    sm_alive_inst.fetch_add(1, std::memory_order_relaxed);
}

ExecutionMask::~ExecutionMask() {
    sm_alive_inst.fetch_sub(1, std::memory_order_relaxed);
}

void ExecutionMask::register_to_opr(OperatorNodeBase* opr) {
    auto&& acc = opr->node_prop().attribute().accessory;
    if (m_owner) {
        mgb_assert(m_owner->owner_graph() == opr->owner_graph());
    }
    mgb_assert(!acc.exec_mask,
               "multiple ExecutionMask objects registered to %s{%s}",
               opr->cname(), opr->dyn_typeinfo()->name);
    acc.exec_mask = this;
    RefHolder::get(opr->owner_graph()).add(shared_from_this());
#if MGB_ENABLE_JSON
    (*opr->to_json_extra_json)["execution_mask"] = json::NumberInt::make(m_id);
#endif
    // require all vars to use dynamic mem since this opr may be disabled
    for (auto i : opr->output()) {
        i->add_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC);
    }
}

void ExecutionMask::enable(bool flag) {
    m_enabled = flag;
    if (!flag && !m_nested.empty()) {
        SmallVector<ExecutionMask*> stack{this};
        while (!stack.empty()) {
            auto cur = stack.back();
            stack.pop_back();
            for (auto i : cur->m_nested) {
                i->m_enabled = false;
                for (auto j : i->m_nested) {
                    stack.emplace_back(j);
                }
            }
        }
    }
}

void ExecutionMask::add_nested(ExecutionMask* nested) {
    mgb_assert(!nested->m_parent && nested->m_nested.empty());
    nested->m_parent = this;
    nested->m_level = m_level + 1;
    m_nested.emplace_back(nested);
}

ExecutionMask* ExecutionMask::find_direct_lowest(ExecutionMask* a,
                                                 ExecutionMask* b) {
    if (!a || a == b) {
        return b;
    }
    if (!b) {
        return a;
    }
    auto ret = b;

    if (a->m_level > b->m_level) {
        std::swap(a, b);
    }

    // check if a is an ancestor of b
    while (b->m_level > a->m_level) {
        if (a == b) {
            return ret;
        }
        b = b->m_parent;
    }
    return nullptr;
}

#endif  // MGB_ENABLE_COND_EXEC

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
