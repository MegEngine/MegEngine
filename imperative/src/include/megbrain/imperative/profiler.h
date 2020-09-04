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

#include <variant>

#include "megbrain/comp_node.h"
#include "megbrain/graph/event.h"
#include "megbrain/utils/json.h"
#include "megbrain/utils/timer.h"

#include "megbrain/imperative/op_def.h"

#include "megbrain/imperative/function_hook.h"

namespace mgb {
namespace imperative {

struct ProfileEntry{
    using TimeClosure = std::function<double()>;
    std::shared_ptr<OpDef> op;
    std::tuple<double, double> host;
    std::vector<std::tuple<CompNode, TimeClosure, TimeClosure>> device_list;
    void wait_device(){
        for(auto& [cn, begin, end]: device_list){
            MGB_MARK_USED_VAR(cn);
            begin = [begin=begin()]{ return begin; };
            end  = [end = end()]{ return end; };
        }
    }
};

using Profile = std::vector<ProfileEntry>;

class DeviceTimer {
public:
    using SharedEvent = std::shared_ptr<CompNode::Event>;
    DeviceTimer() = default;
    void reset(thin_function<double()> host_timer);
    thin_function<double()> get_device_time(CompNode device);

private:
    CompNode::UnorderedMap<std::tuple<SharedEvent, double>> m_base_event_table;
};

class Profiler {
public:
    Profiler(Profile* profile = nullptr) {
        if (!profile) {
            m_owned_profile = std::make_unique<Profile>();
            profile = m_owned_profile.get();
        }
        m_profile = profile;
    }
    void start();
    void stop();
    Profile& get_profile() { return *m_profile; }

private:
    DeviceTimer m_device_timer;
    RealTimer m_host_timer;
    Profile* m_profile;
    std::unique_ptr<Profile> m_owned_profile;
    std::vector<FunctionHooker<decltype(OpDef::apply_on_physical_tensor)>>
            m_hooker_list;
};

}  // namespace imperative
}  // namespace mgb
