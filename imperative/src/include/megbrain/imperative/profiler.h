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

using HostTime = std::chrono::system_clock::time_point;

using Duration = std::chrono::system_clock::duration;

using RealDuration = std::chrono::duration<Duration::rep, Duration::period>;

using Time = HostTime;

}  // namespace profiler

class Timer {
public:
    using Time = profiler::Time;
    static profiler::Time record_host();
    static std::shared_ptr<CompNode::Event> record_device(CompNode device);
};

class AnyPtr {
public:
    struct Deleter {
        void* object;
        void (*method)(void*, void*);
        void operator() (void* ptr) {
            method(object, ptr);
        }
    };
private:
    using holder_t = std::unique_ptr<void, Deleter>;

    const std::type_info* m_type = nullptr;
    holder_t m_holder = nullptr;
public:
    AnyPtr() = default;
    template <typename T, typename=std::enable_if_t<!std::is_same_v<std::decay_t<T>, AnyPtr>>>
    explicit AnyPtr(T* value, Deleter deleter) {
        m_type = &typeid(T);
        m_holder = {value, deleter};
    }
    template <typename T>
    T* as() {
        mgb_assert(is_exactly<T>(), "type mismatch");
        return reinterpret_cast<T*>(m_holder.get());
    }
    template <typename T>
    const T* as() const {
        mgb_assert(is_exactly<T>(), "type mismatch");
        return reinterpret_cast<const T*>(m_holder.get());
    }
    template <typename T>
    bool is_exactly() const {
        return std::type_index{typeid(T)} == std::type_index{*m_type};
    }
    const std::type_info& type() const {
        return *m_type;
    }
    bool operator==(std::nullptr_t nptr) const {
        return m_holder == nullptr;
    }
    operator bool() const {
        return m_holder != nullptr;
    }
};


class Profiler {
public:
    struct Record {
        uint64_t id;
        std::thread::id tid;
        profiler::Time time;
        AnyPtr data;
        Record() = default;
        Record(uint64_t id, std::thread::id tid, profiler::Time time, AnyPtr data):
            id{id}, tid{tid}, time{time}, data{std::move(data)} {};
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
    std::atomic<Status> m_status = Running;
    std::unordered_map<std::type_index, AnyPtr> m_mem_pools;

    static std::vector<entry_t> sm_records;
    static options_t sm_profile_options;
    static std::mutex sm_mutex;
    // assume std::thread::id is unique
    static std::unordered_map<std::thread::id, std::unique_ptr<Profiler>> sm_profilers;
    static Timer sm_timer;
    static profiler::HostTime sm_start_at;
    static std::atomic_uint64_t sm_last_id;
    static std::atomic_size_t sm_preferred_capacity;
    static bool sm_profiling;
    static constexpr bool sm_debug = false;
    thread_local static Profiler* tm_profiler;
public:
    explicit Profiler(std::thread::id tid): m_thread_id{tid} {
        mgb_assert(tid == std::this_thread::get_id(), "thread id mismatch");
    }
public:
    static Profiler& get_instance() {
        if (!tm_profiler) {
            MGB_LOCK_GUARD(sm_mutex);
            auto& profiler = sm_profilers[std::this_thread::get_id()];
            if (!profiler) {
                profiler = std::make_unique<Profiler>(std::this_thread::get_id());
            }
            tm_profiler = profiler.get();
        }
        return *tm_profiler;
    }

    template <typename T>
    static MemPool<T>& get_mem_pool() {
        thread_local MemPool<T>* t_pool = nullptr;
        if (t_pool == nullptr) {
            auto& pool = get_instance().m_mem_pools[typeid(MemPool<T>)];
            if (pool == nullptr) {
                pool = AnyPtr(new MemPool<T>(), {nullptr, [](void*, void* ptr){
                    delete reinterpret_cast<MemPool<T>*>(ptr);
                }});
            }
            t_pool = pool.as<MemPool<T>>();
        }
        return *t_pool;
    }

    static uint64_t next_id() {
        return sm_last_id++;
    }

    template <typename T, typename... TArgs>
    static uint64_t record(TArgs&&... args) {
        auto& profiler = get_instance();
        auto& mem_pool = get_mem_pool<T>();
        if constexpr (sm_debug) {
            Status expected = Running;
            mgb_assert(profiler.m_status.compare_exchange_strong(expected, Recording));
        }
        uint64_t id = next_id();
        profiler::Time time = sm_timer.record_host();
        auto deleter = [](void* obj, void* ptr){
            reinterpret_cast<MemPool<T>*>(obj)->free(reinterpret_cast<T*>(ptr));
        };
        profiler.m_records.emplace_back(id, profiler.m_thread_id, time, AnyPtr{
            mem_pool.alloc(T{std::forward<TArgs>(args)...}), {&mem_pool, deleter}
        });
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
        std::vector<entry_t> profile_data = std::move(sm_records);
        for (auto&& [tid, profiler]: sm_profilers) {
            sm_preferred_capacity = std::max(sm_preferred_capacity.load(), profiler->m_records.size());
            profile_data.insert(profile_data.end(),
                    std::make_move_iterator(profiler->m_records.begin()),
                    std::make_move_iterator(profiler->m_records.end()));
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
        bundle.entries = std::move(profile_data);
        bundle.options = get_options();
        bundle.start_at = sm_start_at;
        bundle.thread_dict = get_thread_dict();
        return std::move(bundle);
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


#define MGB_RECORD_EVENT(type, ...) \
    if (mgb::imperative::Profiler::is_profiling()) { \
        mgb::imperative::Profiler::record<type>(type{__VA_ARGS__}); \
    } \

#define MGB_RECORD_EVENT_IF(expr, type, ...) \
    if (mgb::imperative::Profiler::is_profiling() && (expr)) { \
        mgb::imperative::Profiler::record<type>(type{__VA_ARGS__}); \
    } \

}  // namespace imperative
}  // namespace mgb
