/**
 * \file imperative/src/include/megbrain/imperative/profiler.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <optional>
#include <map>
#include <variant>
#include <fstream>
#include <chrono>
#include <bitset>
#include <deque>
#include <any>
#include <typeindex>

#include "megbrain/comp_node.h"
#include "megbrain/graph/event.h"
#include "megbrain/utils/json.h"
#include "megbrain/utils/timer.h"

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/physical_tensor.h"

namespace mgb {
namespace imperative {

namespace profiler {

using HostTime = std::chrono::time_point<std::chrono::high_resolution_clock>;

using Duration = std::chrono::nanoseconds;
using RealDuration = std::chrono::duration<double, std::nano>;

using Time = HostTime;

}  // namespace profiler

class Timer {
public:
    using Time = profiler::Time;
    static profiler::Time record_host();
    static std::shared_ptr<CompNode::Event> record_device(CompNode device);
};


class Profiler {
public:
    struct Record {
        uint64_t id;
        std::thread::id tid;
        profiler::Time time;
        std::any data;
    };
    enum Status: uint8_t {
        Running = 0,
        Recording = 1,
        Collecting = 2,
    };
    struct ResultBundle;
    using ProfileCollector = std::function<void(Record)>;
    using option_t = uint64_t;
    using options_t = std::unordered_map<std::string, option_t>;
    using entry_t = Record;
    using bundle_t = ResultBundle;
    using thread_dict_t = std::unordered_map<std::thread::id, std::string>;

    struct ResultBundle {
        profiler::HostTime start_at;
        thread_dict_t thread_dict;
        options_t options;
        std::vector<entry_t> entries;
    };
private:
    std::thread::id m_thread_id;
    std::vector<Record> m_records;
    std::vector<std::any> m_duration_stack;
    std::atomic<Status> m_status = Running;
    std::string m_thread_name;

    static options_t sm_profile_options;
    static std::mutex sm_mutex;
    static std::unordered_map<std::thread::id, Profiler*> sm_profilers;
    static Timer sm_timer;
    static profiler::HostTime sm_start_at;
    static std::atomic_uint64_t sm_last_id;
    static std::atomic_size_t sm_preferred_capacity;
    static bool sm_profiling;
    static constexpr bool sm_debug = false;
    thread_local static std::unique_ptr<Profiler> tm_profiler;
public:
    Profiler() {
        m_thread_id = std::this_thread::get_id();
        MGB_LOCK_GUARD(sm_mutex);
        if (sm_profilers.size() == 0) {
            reset();
        }
        mgb_assert(sm_profilers.count(m_thread_id) == 0);
        sm_profilers[m_thread_id] = this;
    }
    ~Profiler() {
        MGB_LOCK_GUARD(sm_mutex);
        mgb_assert(sm_profilers.count(m_thread_id) == 1);
        sm_profilers.erase(m_thread_id);
    }
public:
    static Profiler& get_instance() {
        return *tm_profiler;
    }

    static void reset() {
        mgb_assert(sm_profilers.size() == 0, "profiler already running");
        sm_start_at = profiler::HostTime::min();
    }

    static uint64_t next_id() {
        return sm_last_id++;
    }

    template <typename T, typename... TArgs>
    static uint64_t record(TArgs&&... args) {
        auto& profiler = get_instance();
        if constexpr (sm_debug) {
            Status expected = Running;
            mgb_assert(profiler.m_status.compare_exchange_strong(expected, Recording));
        }
        uint64_t id = next_id();
        profiler::Time time = sm_timer.record_host();
        profiler.m_records.push_back({id, std::this_thread::get_id(), time, T{std::forward<TArgs>(args)...}});
        if constexpr (sm_debug) {
            Status expected = Recording;
            mgb_assert(profiler.m_status.compare_exchange_strong(expected, Running));
        }
        return id;
    }

    static bundle_t collect() {
        bundle_t bundle;
        MGB_LOCK_GUARD(sm_mutex);
        if constexpr (sm_debug) {
            for (auto&& [tid, profiler]: sm_profilers) {
                MGB_MARK_USED_VAR(tid);
                Status expected = Running;
                mgb_assert(profiler->m_status.compare_exchange_strong(expected, Collecting));
            }
        }
        std::vector<entry_t> profile_data;
        for (auto&& [tid, profiler]: sm_profilers) {
            sm_preferred_capacity = std::max(sm_preferred_capacity.load(), profiler->m_records.size());
            for (auto& record: profiler->m_records) {
                profile_data.push_back(std::move(record));
            }
            profiler->m_records.clear();
            profiler->m_records.reserve(sm_preferred_capacity);
        }
        std::sort(profile_data.begin(), profile_data.end(), [](auto& lhs, auto& rhs){
            return lhs.id < rhs.id;
        });
        if constexpr (sm_debug) {
            for (auto&& [tid, profiler]: sm_profilers) {
                MGB_MARK_USED_VAR(tid);
                Status expected = Collecting;
                mgb_assert(profiler->m_status.compare_exchange_strong(expected, Running));
            }
        }
        bundle.entries = profile_data;
        bundle.options = get_options();
        bundle.start_at = sm_start_at;
        bundle.thread_dict = get_thread_dict();
        return bundle;
    }

    static option_t get_option(std::string key, option_t default_val) {
        if (!sm_profile_options.count(key)) {
            return default_val;
        }
        return sm_profile_options.at(key);
    }

    static void load_options(options_t options) {
        sm_profile_options = std::move(options);
    }

    static options_t get_options() {
        return sm_profile_options;
    }

    static bool is_profiling() {
        return sm_profiling;
    }

    static void start_profile() {
        mgb_assert(!sm_profiling);
        sm_start_at = Timer::record_host();
        sm_profiling = true;
    }

    static void stop_profile() {
        mgb_assert(sm_profiling);
        sm_profiling = false;
    }

    static thread_dict_t get_thread_dict();

    static void dump_profile(std::string basename, std::string format, bundle_t result);
};


class ProfileDataCollector {
public:
    template <typename T>
    using SubCollector = std::function<void(uint64_t, std::thread::id, uint64_t, T)>;
private:
    std::unordered_map<std::type_index, SubCollector<std::any>> m_collectors;
public:
    template <typename T>
    ProfileDataCollector& handle(SubCollector<T> collector) {
        auto erased = [collector](uint64_t id, std::thread::id tid, uint64_t time, std::any data){
            collector(id, tid, time, std::any_cast<T>(std::move(data)));
        };
        m_collectors[typeid(T)] = erased;
        return *this;
    }
    void operator()(uint64_t id, std::thread::id tid, uint64_t time, std::any event) {
        std::type_index type = event.type();
        if (m_collectors.count(type) == 0) {
            return;
        }
        auto& handler = m_collectors.at(type);
        handler(id, tid, time, std::move(event));
    }
};

}  // namespace imperative
}  // namespace mgb
