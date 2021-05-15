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

#include "./profiler/formats.h"

namespace mgb {
namespace imperative {

uint64_t Timer::get_nsecs() {
    using namespace std::chrono;
    auto finish = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish - m_start);
    return duration.count();
}

uint64_t Timer::get_started_at() {
    return m_started_at;
}

void Timer::reset() {
    using namespace std::chrono;
    m_start = steady_clock::now();
    auto now_ns = duration_cast<nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    m_started_at = now_ns.count();
}

std::shared_ptr<CompNode::Event> Timer::record_event(CompNode device) {
    auto event = EventPool::with_timer().alloc_shared(device);
    event->record();
    return event;
}

Profiler::options_t Profiler::sm_profile_options;
std::mutex Profiler::sm_mutex;
std::unordered_map<std::thread::id, Profiler*> Profiler::sm_profilers;
Timer Profiler::sm_timer;
std::atomic_uint64_t Profiler::sm_last_id = 0;
bool Profiler::sm_profiling = false;
thread_local std::unique_ptr<Profiler> Profiler::tm_profiler = std::make_unique<Profiler>();
std::atomic_size_t Profiler::sm_preferred_capacity;

auto Profiler::get_thread_dict() -> thread_dict_t {
    MGB_LOCK_GUARD(sm_mutex);
    thread_dict_t thread_dict;
    for (auto&& [tid, profiler]: sm_profilers) {
        thread_dict[tid] = profiler->m_thread_name;
    }
    return thread_dict;
}

void Profiler::dump_profile(std::string basename, std::string format, results_t results, options_t options) {
    auto thread_dict = get_thread_dict();
    if (format == "chrome_timeline.json") {
        profiler::dump_chrome_timeline(basename, options, thread_dict, results);
    } else {
        mgb_log_error("unsupported profiling format %s", format.c_str());
    }
}

}  // namespace imperative

}  // namespace mgb
