/**
 * \file imperative/src/impl/profiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/profiler.h"

#include <variant>

#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/physical_tensor.h"

#include "./event_pool.h"
#include "./op_trait.h"

namespace mgb {

namespace imperative {

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

void DeviceTimer::reset(thin_function<double()> host_timer) {
    CompNode::foreach ([this, host_timer](CompNode device) {
        auto base_event = EventPool::with_timer().alloc_shared(device);
        base_event->record();
        m_base_event_table[device] = {std::move(base_event), host_timer()};
    });
}

thin_function<double()> DeviceTimer::get_device_time(CompNode device) {
    auto event = EventPool::with_timer().alloc_shared(device);
    event->record();
    auto base = m_base_event_table[device];
    return [base, event] {
        auto [base_event, host_time] = base;
        //TODO: sync once for each compnode
        event->host_wait();
        return base_event->elapsed_time_until(*event) * 1000 + host_time;
    };
}

void Profiler::start() {
    m_host_timer.reset();
    m_device_timer.reset([&]{ return m_host_timer.get_msecs();} );
    OpTrait::for_each_trait([this](OpTrait& trait) {
        FunctionHooker hooker{&trait.apply_on_physical_tensor};
        hooker.apply_hook([this](auto&& apply, const OpDef& def,
                                 const SmallVector<TensorPtr>& inputs) {
            ProfileEntry entry;
            entry.op = def.copy();
            double host_begin = m_host_timer.get_msecs();
            auto&& comp_nodes = collect_comp_nodes(def, inputs);
            for (auto&& comp_node : comp_nodes) {
                entry.device_list.push_back(
                        {comp_node,
                         m_device_timer.get_device_time(comp_node),
                         {}});
            }
            auto outputs = apply(def, inputs);
            for (auto& [cn, dev_begin, dev_end] : entry.device_list) {
                MGB_MARK_USED_VAR(cn);
                MGB_MARK_USED_VAR(dev_begin);
                dev_end = m_device_timer.get_device_time(cn);
            }
            entry.host = {host_begin, m_host_timer.get_msecs()};
            m_profile->push_back(std::move(entry));
            return outputs;
        });
        m_hooker_list.push_back(std::move(hooker));
    });
}

void Profiler::stop() {
    m_hooker_list.clear();
    for (auto& entry : *m_profile) {
        entry.wait_device();
    }
}

}  // namespace imperative

}  // namespace mgb
