/**
 * \file src/core/impl/imperative/profiler.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/imperative/profiler.h"

#if defined(_MSC_VER) || defined(WIN32)
#include <windows.h>
#define getpid GetCurrentProcessId
#else
#include <sys/unistd.h>
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include <unistd.h>
#endif

#include <variant>

#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/physical_tensor.h"

#include "./op_trait.h"

namespace mgb {

namespace imperative {

class OpDefInfo{
public:
    size_t id;
    std::string name;
};

class ProfilerEntry {
public:
    ProfilerEntry(size_t index, Profiler::EventKind type, std::unique_ptr<CompNode::Event> device)
        : index{index}, type{type}, device{std::move(device)}{
    }
    ProfilerEntry(size_t index, Profiler::EventKind type, double host): index{index}, type{type}, host{host}{
    }
    size_t index;
    Profiler::EventKind type;
    std::unique_ptr<CompNode::Event> device = nullptr;
    double host = 0;
};

class ProfilerPrivate {
public:
    std::vector<OpDefInfo> op_list;
    std::vector<ProfilerEntry> entry_list;
    std::vector<std::unique_ptr<CompNode::Event>> event_list;
    std::vector<std::tuple<OpTrait*, std::unique_ptr<ApplyOnPhysicalTensor>>>
            hook_list;
    ThinHashMap<CompNode, std::tuple<CompNode::Event*, double>>
            comp_node_begin_map;
    ThinHashMap<CompNode, CompNode::Event*> comp_node_end_map;
    RealTimer timer;
    size_t dump_count = 0;
    bool enabled = false;
    std::string path;
};

namespace {
CompNode::UnorderedSet collect_comp_nodes(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    CompNode::UnorderedSet comp_nodes;
    for (auto&& input : inputs) {
        comp_nodes.insert(input->comp_node());
    }
    for (auto&& output_attr : def.infer_output_attrs(def, inputs)) {
        comp_nodes.insert(output_attr.comp_node);
    }
    return comp_nodes;
}
}  // namespace

std::unique_ptr<CompNode::Event> Profiler::create_event(CompNode comp_node){
    auto event = comp_node.create_event(CompNode::Event::NEED_TIMER);
    event->record();
    auto& [begin, time] = m_private->comp_node_begin_map[comp_node];
    if (begin == nullptr) {
        begin = event.get();
        time = m_private->timer.get_msecs();
    }
    return event;
}

double Profiler::get_host_time_now(){
    return m_private->timer.get_msecs();
}

double Profiler::get_device_time(CompNode::Event& event) {
    auto [base_event, host_time] =
            m_private->comp_node_begin_map[event.comp_node()];
    if (base_event == &event) {
        return host_time;
    } else {
        return host_time + base_event->elapsed_time_until(event) * 1000;
    }
}

size_t Profiler::get_dump_count(){
    return m_private->dump_count;
}

Profiler::Profiler() {
    m_private = std::make_unique<ProfilerPrivate>();
}

Profiler::Profiler(const std::string& path): Profiler() {
    m_private->path = path;
}

void Profiler::enable() {
    m_private->enabled = true;
    CompNode::sync_all();
    OpTrait::for_each_trait([this](OpTrait& trait) {
        auto backup = std::make_unique<ApplyOnPhysicalTensor>(
                std::move(trait.apply_on_physical_tensor));
        trait.apply_on_physical_tensor =
                 [this, backup = backup.get()] (
                        const OpDef& def,
                        const SmallVector<TensorPtr>& inputs){
                    size_t index = m_private->op_list.size();
                    std::string name = "[" + std::to_string(index) + "]" + print_op(def);
                    m_private->op_list.push_back({reinterpret_cast<size_t>(&def), name});
                    m_private->entry_list.emplace_back(index, OprBegin, get_host_time_now());
                    auto&& comp_nodes = collect_comp_nodes(def, inputs);
                    for (auto&& comp_node : comp_nodes) {
                        m_private->entry_list.emplace_back(index, OprBegin, create_event(comp_node));
                    }
                    auto output = (*backup)(def, inputs);
                    for (auto&& comp_node : comp_nodes) {
                        m_private->entry_list.emplace_back(index, OprEnd, create_event(comp_node));
                    }
                    m_private->entry_list.emplace_back(index, OprEnd, get_host_time_now());
                    return output;
                };
        m_private->hook_list.push_back({&trait, std::move(backup)});
    });
}

void Profiler::disable() {
    for (auto&& hook : m_private->hook_list) {
        std::get<0>(hook)->apply_on_physical_tensor =
                std::move(*std::get<1>(hook));
    }
    m_private->hook_list.clear();
    m_private->enabled = false;
}

Profiler::~Profiler() {
}

void Profiler::dump(){
    dump(m_private->path);
}

void Profiler::dump(const std::string& path) {
    using namespace json;
    auto obj = json::Object::make();
    if (!(*obj)["traceEvents"]) {
        (*obj)["traceEvents"] = Array::make();
    }
    auto& trace_events = (*obj)["traceEvents"]->cast_final<Array>();
    for (auto&& entry : m_private->entry_list) {
        auto trace_event_ptr = Object::make();
        auto& trace_event = *trace_event_ptr;
        std::string name;
        size_t id;
        int pid;
        std::string tid;
        double ts;
        const char* ph;
        name = m_private->op_list[entry.index].name;
        id = entry.index;
        pid = getpid();
        if (entry.device) {
            entry.device->host_wait();
            ts = get_device_time(*entry.device);
            tid = entry.device->comp_node().to_string();
        } else {
            ts = entry.host;
            tid = "host";
        }
        switch (entry.type) {
            case OprBegin: {
                ph = "B";
                break;
            }
            case OprEnd: {
                ph = "E";
                break;
            }
        }
        trace_event["name"] = String::make(name);
        trace_event["id"] = Number::make(id);
        trace_event["pid"] = Number::make(pid);
        trace_event["tid"] = String::make(tid);
        trace_event["ts"] = Number::make(ts * 1000);
        trace_event["ph"] = String::make(ph);
        trace_events.add(std::move(trace_event_ptr));
    }
    obj->writeto_fpath(path.empty() ? path : m_private->path);
    m_private->dump_count++;
}

std::string Profiler::print_op(const OpDef& def){
    auto* opr_attr = def.try_cast_final<const OprAttr>();
    if(opr_attr){
        return std::string("OprAttr:") + opr_attr->type;
    }
    return def.dyn_typeinfo()->name;
}

}  // namespace imperative

}  // namespace mgb
