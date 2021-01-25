/**
 * \file imperative/src/impl/profiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/profiler.h"

#include <chrono>

#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/physical_tensor.h"

#include "megbrain/plugin/opr_footprint.h"

#include "./function_hook.h"
#include "./event_pool.h"
#include "./op_trait.h"

namespace mgb {
namespace imperative {

namespace {

DeviceTimer::SharedEvent alloc_recorded_event(CompNode device) {
    auto event = EventPool::with_timer().alloc_shared(device);
    event->record();
    return event;
}

}  // namespace

DeviceTimer::SharedEvent DeviceTimer::get_device_time(CompNode device) {
    return alloc_recorded_event(device);
}

SmallVector<DeviceTimer::SharedEvent> DeviceTimer::get_all(SmallVector<CompNode> device_list) {
    SmallVector<DeviceTimer::SharedEvent> results;
    for (auto&& device: device_list) {
        results.push_back(alloc_recorded_event(device));
    }
    return results;
}

double HostTimer::get_msecs() {
    using namespace std::chrono;
    auto finish = steady_clock::now();
    auto duration = duration_cast<microseconds>(finish - m_start);
    return (double)duration.count() / 1e3;
}

double HostTimer::get_started_at() {
    return m_started_at;
}

void HostTimer::reset() {
    using namespace std::chrono;
    m_start = steady_clock::now();
    auto now_us = duration_cast<microseconds>(std::chrono::system_clock::now().time_since_epoch());
    m_started_at = (double)(now_us.count()) / 1e3;
}

}  // namespace imperative

}  // namespace mgb
