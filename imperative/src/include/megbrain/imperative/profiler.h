/**
 * \file imperative/src/include/megbrain/imperative/profiler.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <any>
#include <optional>
#include <stack>
#include <list>

#include "megbrain/comp_node.h"
#include "megbrain/graph/event.h"
#include "megbrain/utils/json.h"
#include "megbrain/utils/timer.h"

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/physical_tensor.h"

namespace mgb {
namespace imperative {

using ProfileTensor = std::tuple<size_t, std::vector<size_t>, CompNode>;

struct ProfileEntry {
    using TimeClosure = std::function<double()>;
    size_t id;
    size_t parent;
    std::shared_ptr<OpDef> op;
    //(host_begin, host_end)
    std::tuple<double, double> host;
    //[(device, device_begin, device_end)]
    std::vector<std::tuple<CompNode, TimeClosure, TimeClosure>> device_list;
    std::vector<ProfileTensor> inputs;
    std::vector<ProfileTensor> outputs;
    long long memory = 0;
    long long computation = 0;
#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> param;
#endif
    void wait_device() {
        for (auto& [cn, begin, end] : device_list) {
            MGB_MARK_USED_VAR(cn);
            begin = [begin = begin()] { return begin; };
            end = [end = end()] { return end; };
        }
    }
};

using Profile = std::list<ProfileEntry>;

class DeviceTimer {
public:
    using SharedEvent = std::shared_ptr<CompNode::Event>;
    DeviceTimer() = default;
    void reset(thin_function<double()> host_timer);
    thin_function<double()> get_device_time(CompNode device);
    void clear();

private:
    CompNode::UnorderedMap<std::tuple<SharedEvent, double>> m_base_event_table;
    thin_function<double()> m_host_timer;
};

class TensorRecorder {
private:
    // active tensors
    std::unordered_map<Tensor*, std::tuple<std::weak_ptr<Tensor>, size_t>>
            m_tensor_map;
    size_t m_next_id;

public:
    size_t record_tensor(const TensorPtr& tensor);
    void clear();
};

class Profiler {
public:
    enum Flags {
        PROFILE_FOOTPRINT = 1,
    };

public:
    Profiler() = default;
    // Start profiler by hook OpTrait
    void start(uint32_t flags);
    // Stop profiler and clean environment
    void stop();
    void clear();
    Profile& get_profile();

private:
    DeviceTimer m_device_timer;
    RealTimer m_host_timer;
    Profile m_profile;
    TensorRecorder m_tensor_recorder;
    std::stack<std::tuple<const OpDef*, ProfileEntry*, std::thread::id>>
            m_entry_stack;
    // Hold profile owned by this Profiler
    std::unique_ptr<Profile> m_owned_profile;
    // Hold hooks, cleared when stop
    std::vector<std::any> m_hooker_list;
    size_t m_entry_count = 0;
    Spinlock m_lock;
    std::unordered_map<Tensor*, std::weak_ptr<Tensor>> m_recorded_tensors;
};

}  // namespace imperative
}  // namespace mgb
