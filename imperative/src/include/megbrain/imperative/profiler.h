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

class Timer {
public:
    void reset();
    uint64_t get_nsecs();
    uint64_t get_started_at();
    static std::shared_ptr<CompNode::Event> record_event(CompNode device);
private:
    decltype(std::chrono::steady_clock::now()) m_start;
    uint64_t m_started_at;
};


class Profiler {
public:
    struct Record {
        uint64_t id;
        uint64_t time; //in ns
        std::any data;
    };
    enum Status: uint8_t {
        Running = 0,
        Recording = 1,
        Collecting = 2,
    };
    using ProfileCollector = std::function<void(std::thread::id, Record)>;
    using option_t = uint64_t;
    using options_t = std::unordered_map<std::string, option_t>;
    using result_t = std::pair<std::thread::id, Record>;
    using results_t = std::vector<result_t>;
    using thread_dict_t = std::unordered_map<std::thread::id, std::string>;
private:
    std::thread::id m_thread_id;
    std::vector<Record> m_records;
    std::atomic<Status> m_status = Running;
    uint64_t m_last_time = 0;
    std::string m_thread_name;

    static options_t sm_profile_options;
    static std::mutex sm_mutex;
    static std::unordered_map<std::thread::id, Profiler*> sm_profilers;
    static Timer sm_timer;
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
        sm_timer.reset();
    }

    static uint64_t next_id() {
        return sm_last_id++;
    }

    template <typename T, typename... TArgs>
    static uint64_t record(TArgs&&... args) {
        auto& profiler = get_instance();
        auto last_time = profiler.m_last_time;
        if constexpr (sm_debug) {
            Status expected = Running;
            mgb_assert(profiler.m_status.compare_exchange_strong(expected, Recording));
        }
        uint64_t id = next_id();
        uint64_t time = sm_timer.get_nsecs();
        time = std::max(time, last_time + 2000);
        profiler.m_last_time = time;
        profiler.m_records.push_back({id, time, T{std::forward<TArgs>(args)...}});
        if constexpr (sm_debug) {
            Status expected = Recording;
            mgb_assert(profiler.m_status.compare_exchange_strong(expected, Running));
        }
        return id;
    }

    static results_t collect() {
        MGB_LOCK_GUARD(sm_mutex);
        if constexpr (sm_debug) {
            for (auto&& [tid, profiler]: sm_profilers) {
                MGB_MARK_USED_VAR(tid);
                Status expected = Running;
                mgb_assert(profiler->m_status.compare_exchange_strong(expected, Collecting));
            }
        }
        std::vector<std::pair<std::thread::id, Record>> profile_data;
        for (auto&& [tid, profiler]: sm_profilers) {
            sm_preferred_capacity = std::max(sm_preferred_capacity.load(), profiler->m_records.size());
            for (auto& record: profiler->m_records) {
                profile_data.push_back({tid, std::move(record)});
            }
            profiler->m_records.clear();
            profiler->m_records.reserve(sm_preferred_capacity);
        }
        std::sort(profile_data.begin(), profile_data.end(), [](auto& lhs, auto& rhs){
            return lhs.second.id < rhs.second.id;
        });
        if constexpr (sm_debug) {
            for (auto&& [tid, profiler]: sm_profilers) {
                MGB_MARK_USED_VAR(tid);
                Status expected = Collecting;
                mgb_assert(profiler->m_status.compare_exchange_strong(expected, Running));
            }
        }
        return profile_data;
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
        sm_profiling = true;
    }

    static void stop_profile() {
        mgb_assert(sm_profiling);
        sm_profiling = false;
    }

    static thread_dict_t get_thread_dict();

    static void dump_profile(std::string basename, std::string format, results_t results, options_t options);
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
