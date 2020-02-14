/**
 * \file src/plugin/impl/profiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/plugin/profiler.h"
#include "megbrain/plugin/opr_footprint.h"

#if MGB_ENABLE_JSON
#include "megbrain/graph/event.h"
#include "megbrain/opr/io.h"
#include "megbrain/system.h"

using namespace mgb;
using namespace cg;

MGB_TYPEINFO_OBJ_IMPL(opr_profile::OprProfileHolder);

GraphProfiler::GraphProfiler(cg::ComputingGraph* graph) : PluginBase(graph) {
    graph->options()
            .user_data.get_user_data_or_create<opr_profile::OprProfileHolder>();

    using namespace cg::event;
    auto on_seq_start = [this](CompSeqExecBeforeStart const& event) {
        m_used_comp_node = event.used_comp_node;
    };
    auto on_opr_start = [this](OprExecStart const& event) {
        ensure_start_time();
        if (!opr_filter(event.opr))
            return;

        OperatorNodeBase* opr = event.opr;
        for (auto&& comp_node : get_opr_comp_node_set(event.opr)) {
            auto runner = [this, opr, comp_node]() {
                MGB_LOCK_GUARD(m_mtx);

                auto&& hev = m_host_time[{opr, std::this_thread::get_id()}];
                hev.start = m_timer.get_secs();
                hev.kern = -1;

                record_event(m_kern_event[{opr, comp_node}].start, comp_node);
            };
            event.env->dispatch_on_comp_node(comp_node, runner);
        }
    };
    auto on_opr_finish = [this](OprExecFinished const& event) {
        OperatorNodeBase* opr = event.opr;

        if (!opr_filter(opr))
            return;

        for (auto&& comp_node : get_opr_comp_node_set(event.opr)) {
            auto runner = [this, opr]() {
                MGB_LOCK_GUARD(m_mtx);
                m_host_time[{opr, std::this_thread::get_id()}].end =
                        m_timer.get_secs();
            };
            event.env->dispatch_on_comp_node(comp_node, runner);
        }
    };
    auto on_before_kern = [this](BeforeKernel const& event) {
        if (!opr_filter(event.opr))
            return;

        auto footprint = m_opr_footprint_ptr->calc_footprint(event.opr);
        CompNodeEventPtr* evptr;
        {
            MGB_LOCK_GUARD(m_mtx);
            m_opr_fp_rst.emplace(event.opr, footprint);
            auto&& hev = m_host_time[{event.opr, std::this_thread::get_id()}];
            if (hev.kern == -1) {
                hev.kern = m_timer.get_secs();
            }
            evptr = &m_kern_event[{event.opr, event.comp_node}].kern;
        }

        record_event(*evptr, event.comp_node);
    };
    auto on_after_kern = [this](AfterKernel const& event) {
        if (!opr_filter(event.opr))
            return;

        CompNodeEventPtr* evptr;
        {
            MGB_LOCK_GUARD(m_mtx);
            evptr = &m_kern_event[{event.opr, event.comp_node}].end;
        }
        record_event(*evptr, event.comp_node);
    };
    auto on_graph_compile = [this](const CompSeqOrderDetermined&) {
        // clear status after graph recompilation
        m_host_time.clear();
        m_kern_event.clear();
        m_opr_fp_rst.clear();
        m_start_of_time = None;
    };
    auto&& ev = graph->event();
    add_event_handler(
            ev.register_receiver<CompSeqExecBeforeStart>(on_seq_start));
    add_event_handler(ev.register_receiver<OprExecStart>(on_opr_start));
    add_event_handler(ev.register_receiver<OprExecFinished>(on_opr_finish));
    add_event_handler(ev.register_receiver<BeforeKernel>(on_before_kern));
    add_event_handler(ev.register_receiver<AfterKernel>(on_after_kern));
    add_event_handler(
            ev.register_receiver<CompSeqOrderDetermined>(on_graph_compile));
}

GraphProfiler::~GraphProfiler() noexcept {
    auto wait = [](const CompNodeEventPtr& ev) {
        if (ev)
            ev->host_wait();
    };
    for (auto&& i : m_kern_event) {
        wait(i.second.start);
        wait(i.second.kern);
        wait(i.second.end);
    }

    m_owner_graph->options()
            .user_data.pop_user_data<opr_profile::OprProfileHolder>();
}

void GraphProfiler::ensure_start_time() {
    if (!m_start_of_time.valid()) {
        // set up for the first time
        m_start_of_time =
                CompNode::UnorderedMap<std::unique_ptr<CompNode::Event>>();

        for (auto i: *m_used_comp_node) {
            i.sync();
            auto&& event = m_start_of_time.val()[i];
            event = i.create_event(CompNode::Event::NEED_TIMER);
            event->record();
        }
    }
}

void GraphProfiler::record_event(CompNodeEventPtr& dest, CompNode comp_node) {
    if (!dest)
        dest = comp_node.create_event(CompNode::Event::NEED_TIMER);
    dest->record();
}

bool GraphProfiler::opr_filter(cg::OperatorNodeBase* opr) {
    static bool only_wait = MGB_GETENV("MGB_PROFILE_ONLY_WAIT");
    if (!only_wait)
        return true;
    if (!opr->input_waiting_spec().empty())
        return true;
    auto type = opr->dyn_typeinfo();
    return type == opr::Copy::typeinfo() ||
           type == opr::Host2DeviceCopy::typeinfo();
}

std::shared_ptr<json::Object> GraphProfiler::to_json() const {
    using namespace json;
    auto dev_prof = Object::make();

    auto visit_json_obj = [](Object& obj, const std::string& key) -> Object& {
        auto&& v = obj[key];
        if (!v)
            v = Object::make();
        return *static_cast<Object*>(v.get());
    };

    for (auto&& kern_ev : m_kern_event) {
        auto&& opr_prof =
                visit_json_obj(*dev_prof, kern_ev.first.first->id_str());
        auto comp_node = kern_ev.first.second;
        auto&& event = kern_ev.second;
        auto&& start = m_start_of_time->at(comp_node);
        event.end->host_wait();
        opr_prof[comp_node.to_string()] = Object::make({
                {"start",
                 Number::make(start->elapsed_time_until(*event.start))},
                {"kern", Number::make(start->elapsed_time_until(*event.kern))},
                {"end", Number::make(start->elapsed_time_until(*event.end))},
        });
    }

    auto host_prof = Object::make();
    for (auto&& tpair : m_host_time) {
        auto&& opr_prof =
                visit_json_obj(*host_prof, tpair.first.first->id_str());
        auto&& ev = tpair.second;
        opr_prof[sys::get_thread_name(tpair.first.second)] =
                Object::make({{"start", Number::make(ev.start)},
                              {"kern", Number::make(ev.kern)},
                              {"end", Number::make(ev.end)}});
    }

    auto opr_fp = Object::make();
    for (auto&& tpair : m_opr_fp_rst) {
        auto&& opr_fp_item = *static_cast<Object*>(opr_fp.get());
        opr_fp_item[tpair.first->id_str()] = tpair.second.to_json();
    }

    auto pf_holder_pair =
            m_owner_graph->options()
                    .user_data.get_user_data<opr_profile::OprProfileHolder>();
    mgb_assert(pf_holder_pair.second, "UserData OprProfileHolder not exist.");
    auto opr_internal_pf = Object::make();
    if ((pf_holder_pair.first[0]->id2object_map).size()) {
        for (auto&& pf_pair : pf_holder_pair.first[0]->id2object_map) {
            auto&& opr_itnl_pf_item =
                    *static_cast<Object*>(opr_internal_pf.get());
            opr_itnl_pf_item[pf_pair.first->id_str()] = pf_pair.second;
        }
    }
    return Object::make({{"device", dev_prof},
                         {"host", host_prof},
                         {"opr_footprint", opr_fp},
                         {"opr_internal_pf", opr_internal_pf}});
}

#endif  // MGB_ENABLE_JSON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
