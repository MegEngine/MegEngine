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
#include <unordered_map>

#include "megbrain/imperative/cpp_cupti.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/physical_tensor.h"

#include "megbrain/plugin/opr_footprint.h"

#include "./event_pool.h"
#include "./function_hook.h"
#include "./op_trait.h"

#include "./profiler/formats.h"

namespace mgb {
namespace imperative {

profiler::Time Timer::record_host() {
    return std::chrono::system_clock::now();
}

std::shared_ptr<CompNode::Event> Timer::record_device(CompNode device) {
    auto event = EventPool::with_timer().alloc_shared(device);
    event->record();
    return event;
}

std::vector<Profiler::entry_t> Profiler::sm_records;
Profiler::options_t Profiler::sm_profile_options;
std::mutex Profiler::sm_mutex;
std::unordered_map<std::thread::id, std::unique_ptr<Profiler>> Profiler::sm_profilers;
Timer Profiler::sm_timer;
profiler::HostTime Profiler::sm_start_at = profiler::HostTime::min();
std::atomic_uint64_t Profiler::sm_last_id = 0;
bool Profiler::sm_profiling = false;
thread_local Profiler* Profiler::tm_profiler = nullptr;
std::atomic_size_t Profiler::sm_preferred_capacity;

void Profiler::start_profile() {
    mgb_assert(!sm_profiling);
    sm_start_at = Timer::record_host();
    sm_profiling = true;
    if (cupti::enabled()) {
        MGB_RECORD_EVENT(profiler::CUPTITimestampEvent, cupti::clock::now());
    }
}

void Profiler::stop_profile() {
    mgb_assert(sm_profiling);
    cupti::flush();
    sm_profiling = false;
}

auto Profiler::get_thread_dict() -> thread_dict_t {
    thread_dict_t thread_dict;
    for (auto&& [tid, profiler] : sm_profilers) {
        thread_dict[tid] = sys::get_thread_name(tid);
    }
    return thread_dict;
}

void Profiler::dump_profile(std::string basename, std::string format, bundle_t result) {
    static std::unordered_map<std::string, void (*)(std::string, bundle_t)>
            format_table = {
                    {"chrome_timeline.json", profiler::dump_chrome_timeline},
                    {"memory_flow.svg", profiler::dump_memory_flow},
            };
    auto iter = format_table.find(format);
    if (iter == format_table.end()) {
        mgb_log_error("unsupported profiling format %s", format.c_str());
    }
    return (iter->second)(basename, std::move(result));
}

}  // namespace imperative

}  // namespace mgb
