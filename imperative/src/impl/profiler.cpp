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

#include "./function_hook.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/physical_tensor.h"

#include "megbrain/plugin/opr_footprint.h"

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

DeviceTimer::SharedEvent alloc_recorded_event(CompNode device) {
    auto event = EventPool::with_timer().alloc_shared(device);
    event->record();
    return event;
}

OprFootprint footprint{};

}  // namespace

void DeviceTimer::reset(thin_function<double()> host_timer) {
    CompNode::foreach ([this, host_timer](CompNode device) {
        m_base_event_table[device] = {alloc_recorded_event(device), host_timer()};
    });
    m_host_timer = host_timer;
}

thin_function<double()> DeviceTimer::get_device_time(CompNode device) {
    auto event = EventPool::with_timer().alloc_shared(device);
    event->record();
    if(m_base_event_table.count(device) == 0) {
        m_base_event_table[device] = {alloc_recorded_event(device), m_host_timer()};
    }
    auto base = m_base_event_table[device];
    return [base, event] {
        auto [base_event, host_time] = base;
        // TODO: sync once for each compnode
        event->host_wait();
        return base_event->elapsed_time_until(*event) * 1000 + host_time;
    };
}

void DeviceTimer::clear() {
    m_base_event_table.clear();
}

size_t TensorRecorder::record_tensor(const TensorPtr& tensor) {
    if (m_tensor_map.count(tensor.get()) > 0) {
        auto& [prev, id] = m_tensor_map[tensor.get()];
        if (prev.lock() != tensor) {
            prev = tensor;
            id = m_next_id++;
        }
        return id;
    } else {
        auto id = m_next_id++;
        m_tensor_map.insert(
                {tensor.get(), {std::weak_ptr<Tensor>{tensor}, id}});
        return id;
    }
}

void TensorRecorder::clear() {
    m_next_id = 0;
    m_tensor_map.clear();
}

Profile& Profiler::get_profile() {
    for (auto& entry : m_profile) {
        for (auto& [device, device_begin, device_end] : entry.device_list) {
            MGB_MARK_USED_VAR(device);
            device_begin = [value = device_begin()] { return value; };
            device_end = [value = device_end()] { return value; };
        }
    }
    return m_profile;
}

void Profiler::start(uint32_t flags) {
    m_host_timer.reset();
    m_device_timer.reset([&] { return m_host_timer.get_msecs(); });
    OpTrait::for_each_trait([this, flags](OpTrait& trait) {
        auto hook_apply_on_physical_tensor =
                make_shared_hook(&trait.apply_on_physical_tensor);
        auto hook_apply_on_var_node =
                make_shared_hook(&trait.apply_on_var_node);
        hook_apply_on_physical_tensor->apply_hook([this, flags]
                (auto&& apply, const OpDef& def, const SmallVector<TensorPtr>& inputs) {
            auto shape2vector = [](const TensorShape& shape) {
                std::vector<size_t> vector_shape;
                for (size_t i = 0; i < shape.ndim; i++) {
                    vector_shape.push_back(shape[i]);
                }
                return vector_shape;
            };
            ProfileEntry entry;
            entry.id = m_entry_count++;
            // TODO: assign parent
            entry.parent = 0;
            // Record apply context and save to m_profile
            entry.op = def.copy();
            for (auto&& input : inputs) {
                entry.inputs.push_back({m_tensor_recorder.record_tensor(input),
                                        shape2vector(input->layout()),
                                        input->comp_node()});
            }
            double host_begin = m_host_timer.get_msecs();
            auto&& comp_nodes = collect_comp_nodes(def, inputs);
            for (auto&& comp_node : comp_nodes) {
                entry.device_list.push_back(
                        {comp_node,
                         m_device_timer.get_device_time(comp_node),
                         {}});
            }
            if (flags & PROFILE_FOOTPRINT) {
                MGB_LOCK_GUARD(m_lock);
                m_entry_stack.push({&def, &entry, std::this_thread::get_id()});
            }
            // Do real apply
            auto outputs = apply(def, inputs);
            for (auto& [cn, dev_begin, dev_end] : entry.device_list) {
                MGB_MARK_USED_VAR(cn);
                MGB_MARK_USED_VAR(dev_begin);
                dev_end = m_device_timer.get_device_time(cn);
            }
            entry.host = {host_begin, m_host_timer.get_msecs()};
            for (auto&& output : outputs) {
                entry.outputs.push_back(
                        {m_tensor_recorder.record_tensor(output),
                         shape2vector(output->layout()), output->comp_node()});
            }
            if (flags & PROFILE_FOOTPRINT) {
                mgb_assert(std::get<1>(m_entry_stack.top()) == &entry);
                MGB_LOCK_GUARD(m_lock);
                m_entry_stack.pop();
            }
            m_profile.push_back(std::move(entry));
            return outputs;
        });
        if (flags & PROFILE_FOOTPRINT) {
            hook_apply_on_var_node->apply_hook(
                    [this](auto&& apply, const OpDef& def,
                           VarNodeArray inputs) -> cg::OperatorNodeBase* {
                        auto* operator_node = apply(def, std::move(inputs));
                        std::remove_reference_t<decltype(m_entry_stack.top())>
                                top;
                        {
                            MGB_LOCK_GUARD(m_lock);
                            if (m_entry_stack.empty()) {
                                return operator_node;
                            }
                            top = m_entry_stack.top();
                        }
                        auto [current_op, current_entry, thread_id] = top;
                        if (current_op != &def ||
                            thread_id != std::this_thread::get_id()) {
                            return operator_node;
                        }
                        auto&& footprint_result =
                                footprint.calc_footprint(operator_node);
                        current_entry->memory = footprint_result.memory;
                        current_entry->computation =
                                footprint_result.computation;
#if MGB_ENABLE_JSON
                        current_entry->param = footprint_result.param;
#endif
                        return operator_node;
                    });
        }
        m_hooker_list.push_back(std::move(hook_apply_on_physical_tensor));
        m_hooker_list.push_back(std::move(hook_apply_on_var_node));
    });
}

void Profiler::stop() {
    m_hooker_list.clear();
    for (auto& entry : m_profile) {
        entry.wait_device();
    }
}

void Profiler::clear() {
    mgb_assert(m_entry_stack.empty(),
               "entry_stack should be empty after profile");
    mgb_assert(m_hooker_list.empty(), "hooks should be released");
    m_profile.clear();
    m_entry_count = 0;
    m_device_timer.clear();
    m_tensor_recorder.clear();
}

}  // namespace imperative

}  // namespace mgb
