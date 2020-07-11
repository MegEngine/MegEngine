/**
 * \file src/core/impl/graph/topo_sort.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/execution_mask.h"
#include "megbrain/graph/helper.h"
#include "megbrain/utils/arith_helper.h"

#include <queue>
#include <tuple>

using namespace mgb;
using namespace cg;

TopoSorter::TopoSorter(ComputingGraphImpl* graph) : m_owner_graph{graph} {}

TopoSorter::~TopoSorter() noexcept = default;

struct TopoSorter::NodeTrait {
    static constexpr size_t NPOS = SIZE_MAX;

    //! whether currently in stack during dfs, for loop detection
    bool in_stack = false;

    //! opr priority, can be modified in priority_remap
    int priority = 0;

    //! position in final BFS sequence, or NPOS if not currently in sequence
    size_t pos = NPOS;

    //! step number in dfs_discover_deps(), or NPOS if not visited
    size_t dfs_step_num = NPOS;

    //! number of oprs that this opr depends on, modified during bfs
    size_t unresolved_dep_cnt = 0;

    //! nodes that depend on this opr
    OprNodeArray receivers;

    //! missing tag handler inputs that have been resolved by after
    //! executing operator (can either be resolved by previous opr or
    //! this opr)
    SharedSet<static_infer::StaticInferManagerImpl::TagHandler*>
            resolved_tag_handlers;
};

struct TopoSorter::State {
    ThinHashMap<OperatorNodeBase*, NodeTrait> opr_trait;
    using OprTraitIter = decltype(opr_trait.begin());

    //! map from src var to updated var
    ThinHashMap<VarNode*, VarNode*> var_force_update_dest;
};

class TopoSorter::DFSDepDiscover {
    using NP = OperatorNodeBase::NodeProp;
    struct StackFrame;
    typedef void (DFSDepDiscover::*Proc)();

    TopoSorter* const m_topo_sorter;
    std::vector<StackFrame> m_stack_buf;
    size_t m_step_num = 0;

    //! current frame
    StackFrame* m_cur_frame;

    //! used for passing return value of callee
    NodeTrait* m_prev_return_trait;

    /*!
     * \brief schedule for dfs on new opr
     * \param resume return address; the function body should start with
     *      add_receiver_to_subcall
     */
    void push_stack(OperatorNodeBase* opr, Proc resume);

    //! pop current frame and set *frame* to previous frame
    void pop_stack();

    //! add opr in current frame to the receiver list of just finished callee
    void add_receiver_to_subcall();

    //! entry point
    void proc_start();

    //! add dev comp order dep in dep_map
    void proc_add_dep_comp_order0();
    void proc_add_dep_comp_order1();

    //! add missing inputs required shape/host_value dep type
    void proc_find_missing_inp();
    void proc_dfs_missing_dep0();
    void proc_dfs_missing_dep1();

    //! post process
    void proc_post();

    //! used as the cont_addr for outermost frame
    void proc_empty() {}

public:
    DFSDepDiscover(TopoSorter* topo_sorter) : m_topo_sorter{topo_sorter} {}

    void add_opr(OperatorNodeBase* endpoint);
};

struct TopoSorter::DFSDepDiscover::StackFrame {
    //! operator whose deps are to be discovered
    OperatorNodeBase* const opr;

    //! node trait associated with the operator
    NodeTrait* const trait;

    //! end for dep_map() of opr
    const OperatorNodeBase::NodeProp::DepMap::const_iterator dep_map_end;
    //! current iter position
    OperatorNodeBase::NodeProp::DepMap::const_iterator dep_map_iter;

    //! extra comp order dep that should be added just before return
    VarNodeArray extra_comp_order_dep_to_add;

    //! missing inputs, setup by proc_find_missing_inp()
    SmallVector<const static_infer::StaticInferManagerImpl::TagHandlerSet*>
            missing_inputs;
    std::pair<size_t, decltype(missing_inputs[0]->begin())> missing_inputs_iter;

    //! execution state to resume when returning control to this frame
    void (DFSDepDiscover::*cont_addr)();

    StackFrame(OperatorNodeBase* opr, NodeTrait* trait)
            : opr{opr},
              trait{trait},
              dep_map_end{opr->node_prop().dep_map().end()} {}

    //! advance missing_inputs_iter to iterate in missing_inputs
    void advance_missing_inputs_iter() {
        auto&& iter = missing_inputs_iter;
        ++iter.second;
        auto&& mi = missing_inputs;
        if (iter.second == mi[iter.first]->end()) {
            ++iter.first;
            iter.second = {};
            if (iter.first < mi.size())
                iter.second = mi[iter.first]->begin();
        }
    }
};

void TopoSorter::DFSDepDiscover::add_opr(OperatorNodeBase* endpoint) {
    m_prev_return_trait = nullptr;
    m_cur_frame = nullptr;

    m_stack_buf.clear();
    push_stack(endpoint, &DFSDepDiscover::proc_empty);
    while (m_cur_frame) {
        auto cont = m_cur_frame->cont_addr;
        (this->*cont)();
    }
}

void TopoSorter::DFSDepDiscover::push_stack(OperatorNodeBase* opr,
                                            Proc resume) {
    auto&& trait = m_topo_sorter->m_state->opr_trait[opr];
    mgb_assert(!trait.in_stack, "circular dep in graph");

    if (trait.dfs_step_num != NodeTrait::NPOS) {
        // return directly if already visited
        m_prev_return_trait = &trait;
        return (this->*resume)();
    }

    auto&& frame = m_cur_frame;
    if (frame)
        frame->cont_addr = resume;
    m_stack_buf.emplace_back(opr, &trait);
    frame = &m_stack_buf.back();
    m_prev_return_trait = nullptr;
    frame->cont_addr = &DFSDepDiscover::proc_start;
}

void TopoSorter::DFSDepDiscover::pop_stack() {
    auto&& frame = m_cur_frame;
    mgb_assert(!m_stack_buf.empty() && frame == &m_stack_buf.back());
    m_prev_return_trait = frame->trait;
    m_stack_buf.pop_back();
    frame = m_stack_buf.empty() ? nullptr : &m_stack_buf.back();
}

void TopoSorter::DFSDepDiscover::add_receiver_to_subcall() {
    auto frame = m_cur_frame;
    auto&& t1 = *m_prev_return_trait;
    t1.receivers.push_back(frame->opr);
    ++frame->trait->unresolved_dep_cnt;
}

void TopoSorter::DFSDepDiscover::proc_start() {
    auto frame = m_cur_frame;
    auto&& trait = *frame->trait;
    mgb_assert(trait.dfs_step_num == NodeTrait::NPOS);
    mgb_assert(frame->opr->owner_graph() == m_topo_sorter->m_owner_graph);
    trait.in_stack = true;
    frame->dep_map_iter = frame->opr->node_prop().dep_map().begin();
    return proc_add_dep_comp_order0();
}

void TopoSorter::DFSDepDiscover::proc_add_dep_comp_order0() {
    /*
     * overall flow for adding comp order dep
     *
     * while (dep_map_iter != dep_map_end) {
     *     proc_add_dep_comp_order0()
     *     dfs()
     *     proc_add_dep_comp_order1()
     *     ++ dep_map_iter
     * }
     */
    auto frame = m_cur_frame;
    auto owner_graph = m_topo_sorter->m_owner_graph;
    for (;;) {
        if (frame->dep_map_iter == frame->dep_map_end) {
            return proc_find_missing_inp();
        }
        auto&& dep_entry = *frame->dep_map_iter;
        mgb_assert(dep_entry.first->owner_graph() == owner_graph);
        if (NP::is_device_comp_order_dep(dep_entry.second)) {
            return push_stack(dep_entry.first->owner_opr(),
                              &DFSDepDiscover::proc_add_dep_comp_order1);
        } else {
            ++frame->dep_map_iter;
        }
    }
}

void TopoSorter::DFSDepDiscover::proc_add_dep_comp_order1() {
    auto frame = m_cur_frame;
    add_receiver_to_subcall();
    auto&& dep_entry = *frame->dep_map_iter;
    auto var = dep_entry.first;
    auto&& trait = *frame->trait;
    trait.resolved_tag_handlers.merge_from(
            m_prev_return_trait->resolved_tag_handlers);

    if (NP::is_device_value_dep(dep_entry.second)) {
        ++m_topo_sorter->m_cur_extra_info->var2recvinfo[var].dev_value;
    }

    ++frame->dep_map_iter;
    proc_add_dep_comp_order0();
}

void TopoSorter::DFSDepDiscover::proc_find_missing_inp() {
    auto frame = m_cur_frame;
    auto opr = frame->opr;
    auto&& mgr = ComputingGraphImpl::downcast(opr->owner_graph())
                         ->static_infer_manager_impl();
    auto&& missing_inp = frame->missing_inputs;

    for (auto&& dep_entry : opr->node_prop().dep_map()) {
        // find deps for host value/shape on which static infer fails

        using DT = OperatorNodeBase::NodeProp::DepType;

        if (dep_entry.second & DT::VALUE_ALLOW_EMPTY) {
            auto&& recv_info = m_topo_sorter->m_cur_extra_info
                                       ->var2recvinfo[dep_entry.first];
            mgb_assert(dep_entry.second & (DT::HOST_VALUE | DT::DEV_VALUE));
            ++recv_info.allow_empty_value;
        }

        // get tag handler if satic infer fails
        static_infer::StaticInferManagerImpl::TagHandler* tag_handler = nullptr;

        auto var = dep_entry.first;

        bool static_inferable;
        if (dep_entry.second & DT::HOST_VALUE) {
            static_inferable = cg::is_static_var_value(var);
            tag_handler = mgr.get_tag_handler_for_value(var);
        } else if (dep_entry.second & DT::SHAPE) {
            static_inferable = cg::is_static_var_shape(var);
            tag_handler = mgr.get_tag_handler_for_shape(var);
        } else {
            continue;
        }

        mgb_assert(tag_handler);
        m_topo_sorter->m_cur_extra_info->infer_dest.insert(tag_handler);
        if (!static_inferable) {
            missing_inp.push_back(&mgr.get_missing_inp(tag_handler));
        }
    }

    if (missing_inp.empty()) {
        frame->missing_inputs_iter = {0, {}};
    } else {
        frame->missing_inputs_iter = {0, missing_inp.front()->begin()};
    }
    proc_dfs_missing_dep0();
}

void TopoSorter::DFSDepDiscover::proc_dfs_missing_dep0() {
    auto frame = m_cur_frame;
    auto&& trait = *frame->trait;
    for (;;) {
        if (frame->missing_inputs_iter.first == frame->missing_inputs.size()) {
            return proc_post();
        }
        auto i = *frame->missing_inputs_iter.second;
        if (trait.resolved_tag_handlers.contain(i)) {
            frame->advance_missing_inputs_iter();
            continue;
        }

        VarNode* ivar = i->tag();
        frame->extra_comp_order_dep_to_add.push_back(ivar);
        return push_stack(ivar->owner_opr(),
                          &DFSDepDiscover::proc_dfs_missing_dep1);
    }
}

void TopoSorter::DFSDepDiscover::proc_dfs_missing_dep1() {
    auto frame = m_cur_frame;
    add_receiver_to_subcall();
    using HT = static_infer::StaticInferManagerImpl::TagHandlerType;
    auto i = *frame->missing_inputs_iter.second;
    VarNode* ivar = i->tag();
    auto&& recv_info = m_topo_sorter->m_cur_extra_info->var2recvinfo[ivar];
    if (i->handler_type() == HT::SHAPE) {
        ++recv_info.shape;
    } else {
        mgb_assert(i->handler_type() == HT::VALUE);
        ++recv_info.host_value;
    }
    frame->trait->resolved_tag_handlers.insert(i);
    frame->advance_missing_inputs_iter();
    proc_dfs_missing_dep0();
}

void TopoSorter::DFSDepDiscover::proc_post() {
    auto frame = m_cur_frame;
    auto&& mgr = m_topo_sorter->m_owner_graph->var_node_mem_manager();
    auto opr = frame->opr;

    // find and record force update pairs
    for (auto dest : opr->output()) {
        auto src = mgr.get_var_node_mem_trait(dest).force_update_src;
        if (!src)
            continue;
        auto ins = m_topo_sorter->m_state->var_force_update_dest.emplace(src,
                                                                         dest);
        if (!ins.second) {
            auto opr0 = ins.first->second->owner_opr();
            MGB_MARK_USED_VAR(opr0);
            mgb_throw(GraphError,
                      "variable %s force updated by two oprs: %s{%s} %s{%s}",
                      src->cname(), opr0->cname(), opr0->dyn_typeinfo()->name,
                      opr->cname(), opr->dyn_typeinfo()->name);
        }
    }

    auto&& trait = frame->trait;
    trait->in_stack = false;
    trait->dfs_step_num = (m_step_num++);
    trait->priority = opr->node_prop().attribute().priority;

    for (auto i : frame->extra_comp_order_dep_to_add)
        m_topo_sorter->add_extra_comp_order_dep(opr, i);

    return pop_stack();
}

const OprNodeArray* TopoSorter::get_comp_seq(CompSeqExtraInfo& extra_info,
                                             const VarNodeArray& dest) {
    // move to temporary var to be exception-safe
    PriorityRemapper priority_remapper;
    if (m_priority_remapper) {
        m_priority_remapper.swap(priority_remapper);
    }

    m_cur_extra_info = &extra_info;
    m_seq.clear();
    auto state = std::make_unique<State>();
    m_state = state.get();
    mgb_assert(m_modified_dep_map_log.empty(), "restore_opr_prop() not called");

    {
        // run the dfs
        DFSDepDiscover dfs{this};
        for (auto i : dest)
            dfs.add_opr(i->owner_opr());
    }

#if MGB_ENABLE_COND_EXEC
    // add dependency due to ExecutionMask (conditional oprs must wait for
    // ExecutionMask owner var to be computed first)
    if (ExecutionMask::have_alive_instance()) {
        for (auto&& i : state->opr_trait) {
            if (auto mask = ExecutionMask::get_from_opr(i.first)) {
                if (auto var = mask->owner()) {
                    state->opr_trait.at(var->owner_opr())
                        .receivers.push_back(i.first);
                    ++i.second.unresolved_dep_cnt;
                    add_extra_comp_order_dep(i.first, var);
                }
            }
        }
    }
#endif

    // add force update control deps
    for (auto&& i : state->var_force_update_dest) {
        auto dest_opr = i.second->owner_opr();
        auto&& dest_trait = state->opr_trait.at(dest_opr);
        for (auto reader : m_owner_graph->var_receiver(i.first)) {
            if (reader == dest_opr)
                continue;
            auto iter = state->opr_trait.find(reader);
            if (iter == state->opr_trait.end())
                continue;

            // reader must finish before dest_opr
            for (auto i : reader->output())
                add_extra_comp_order_dep(dest_opr, i);
            iter->second.receivers.push_back(dest_opr);
            ++dest_trait.unresolved_dep_cnt;
        }
    }

    // remap priority
    if (priority_remapper) {
        auto&& t = m_state->opr_trait;
        std::unique_ptr<PriorityItem[]> items{new PriorityItem[t.size()]};
        size_t idx = 0;
        for (auto&& i : t) {
            mgb_assert(i.second.dfs_step_num < t.size());
            items[idx++] = {i.first, &i.second.priority, i.second.dfs_step_num};
        }
        priority_remapper(dest, items.get(), t.size());
    }

    bfs_make_seq();

    m_cur_extra_info = nullptr;
    m_state = nullptr;
    return &m_seq;
}

class TopoSorter::BFSQueueElem {
    using OprTraitIter = State::OprTraitIter;

    int m_priority;
    size_t m_input_update_time, m_id;
    OprTraitIter m_trait_iter;

public:
    BFSQueueElem() = default;

    BFSQueueElem(TopoSorter* sorter, size_t time, const OprTraitIter& iter)
            : m_input_update_time(time), m_trait_iter(iter) {
        OperatorNodeBase* opr = iter->first;
        m_id = opr->id();
        m_priority = iter->second.priority;

#if MGB_ENABLE_JSON
        {
            // dump extra json
            auto&& json_obj = *opr->to_json_extra_json;
            json_obj["priority"] = json::NumberInt::make(m_priority);
            json_obj["input_update_time"] =
                    json::NumberInt::make(m_input_update_time);
            json_obj["dfs_step_num"] =
                    json::NumberInt::make(iter->second.dfs_step_num);
        }
#endif
    }

    //! whether this element should be placed before rhs
    bool order_before(const BFSQueueElem& rhs) const {
        /*
         * key #0 is priority
         * key #1 is reversed input_update_time, so operator chains could be
         *        executed together
         * key #2 is reversed ID, for stable sorting
         */
        return std::forward_as_tuple(m_priority, rhs.m_input_update_time,
                                     rhs.m_id) <
               std::forward_as_tuple(rhs.m_priority, m_input_update_time, m_id);
    }

    //! used for std::priority_queue
    bool operator<(const BFSQueueElem& rhs) const {
        return rhs.order_before(*this);
    }

    OprTraitIter trait_iter() const { return m_trait_iter; }
};

void TopoSorter::bfs_make_seq() {
    std::priority_queue<BFSQueueElem> boundary_nodes;
    size_t cur_timestamp = 0;
    auto put_queue = [&](State::OprTraitIter node) {
        boundary_nodes.push({this, cur_timestamp, node});
    };
    auto state = m_state;
    for (auto i = state->opr_trait.begin(); i != state->opr_trait.end(); ++i) {
        auto&& t = i->second;
        mgb_assert(t.pos == NodeTrait::NPOS);
        if (!t.unresolved_dep_cnt)
            put_queue(i);
    }
    size_t nr_node_to_add = state->opr_trait.size();

    while (!boundary_nodes.empty()) {
        BFSQueueElem cur = boundary_nodes.top();
        boundary_nodes.pop();
        --nr_node_to_add;

        auto&& node_trait = cur.trait_iter()->second;
        // update node trait
        node_trait.pos = m_seq.size();
        m_seq.push_back(cur.trait_iter()->first);

        ++cur_timestamp;
        for (auto&& other_opr : node_trait.receivers) {
            auto iter = state->opr_trait.find(other_opr);
            mgb_assert(iter != state->opr_trait.end());
            if ((--iter->second.unresolved_dep_cnt) == 0) {
                put_queue(iter);
            }
        }
    }

    if (nr_node_to_add) {
#if MGB_ENABLE_EXCEPTION
        std::string msg{
                "detected circular dependency during topo sort; "
                "this is usually caused by simultaneous reading from a "
                "variable and "
                "its updated version. List of unresolved update var pairs:"};
        for (auto&& i : state->var_force_update_dest) {
            auto v0 = i.first, v1 = i.second;
            if (std::max(state->opr_trait[v0->owner_opr()].pos,
                         state->opr_trait[v1->owner_opr()].pos) ==
                NodeTrait::NPOS) {
                msg.append(ssprintf("\n%s, %s", v0->cname(), v1->cname()));
            }
        }
        mgb_throw_raw(GraphError{msg});
#else
        mgb_trap();
#endif
    }
}

void TopoSorter::add_extra_comp_order_dep(OperatorNodeBase* opr, VarNode* var) {
    auto&& node_prop = const_cast<OprNodeProp&>(opr->node_prop());
    auto&& dep_map = node_prop.dep_map();
    auto iter = dep_map.find(var);
    using DepType = OprNodeProp::DepType;
    DepType orig_v =
            iter == dep_map.end() ? OprNodeProp::DepType{} : iter->second;
    constexpr DepType dt_add = DepType::DEV_COMP_ORDER;
    if (!(orig_v & dt_add)) {
        node_prop.add_dep_type(var, dt_add);
        m_modified_dep_map_log.emplace_back(opr, var, orig_v);
    }
}

void TopoSorter::restore_opr_prop() {
    // iter in reverse order to handle the case when an (opr, var) pair is
    // modified multiple times
    for (auto&& i : reverse_adaptor(m_modified_dep_map_log)) {
        OperatorNodeBase* opr;
        VarNode* var;
        OprNodeProp::DepType dep;
        std::tie(opr, var, dep) = i;

        auto&& dep_map =
                const_cast<OprNodeProp::DepMap&>(opr->node_prop().dep_map());

        if (dep == OprNodeProp::DepType{})
            dep_map.erase(var);
        else
            dep_map[var] = dep;
    }
    m_modified_dep_map_log.clear();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

