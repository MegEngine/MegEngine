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

#include "megbrain/comp_node.h"
#include "megbrain/graph/event.h"
#include "megbrain/utils/json.h"
#include "megbrain/utils/timer.h"

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/physical_tensor.h"

namespace mgb {
namespace imperative {

class DeviceTimer {
public:
    using SharedEvent = std::shared_ptr<CompNode::Event>;
    DeviceTimer() = default;
    SharedEvent get_device_time(CompNode device);
    SmallVector<SharedEvent> get_all(SmallVector<CompNode> device_list);
};

class HostTimer {
public:
    void reset();
    double get_msecs();
    double get_started_at();
private:
    decltype(std::chrono::steady_clock::now()) m_start;
    double m_started_at;
};


class ProfilerBase {
public:
    using Host = std::thread::id;
    using Device = CompNode;

    struct HostInstant {
        Host tid;
        double time;

        void wait() const {}
    };

    struct DeviceInstant {
        double before;
        std::shared_ptr<CompNode::Event> event;
        double after;

        void wait() const {
            event->host_wait();
        }
    };

    using Instant = std::variant<HostInstant, DeviceInstant>;

    template <typename TEvent>
    struct EventRecord {
        Instant instant;
        TEvent data;

        const HostInstant& host() const {
            return std::get<HostInstant>(instant);
        }

        const DeviceInstant& device() const {
            return std::get<DeviceInstant>(instant);
        }

        void wait() const {
            std::visit([&](const auto& instant){ instant.wait(); }, instant);
        }
    };
protected:
    HostInstant record_host() {
        return {std::this_thread::get_id(), m_host_timer.get_msecs()};
    }
    DeviceInstant record_device(Device device) {
        auto before = m_host_timer.get_msecs();
        auto event = m_device_timer.get_device_time(device);
        auto after = m_host_timer.get_msecs();
        return {before, event, after};
    }
protected:
    std::atomic_int64_t m_last_id = 0;
    HostTimer m_host_timer;
    DeviceTimer m_device_timer;
    Spinlock m_lock;
};


template <typename... TEvents>
class Profiler: public ProfilerBase {
public:
    using Record = std::variant<EventRecord<TEvents>...>;
    using Mask = std::bitset<sizeof...(TEvents)>;

    struct Data {
        std::vector<Record> records;
        double started_at;
    };

    template <typename TEvent, size_t index = 0>
    static constexpr size_t index_of() {
        if constexpr (index == std::variant_size_v<Record>) {
            return index;
        } else if constexpr (std::is_same_v<EventRecord<TEvent>, std::variant_alternative_t<index, Record>>) {
            return index;
        } else {
            return index_of<TEvent, index+1>();
        }
    };

    template <typename... TEvents2>
    static Mask mask_of() {
        return Mask{} | (Mask{}.set(index_of<TEvents2>()) |...);
    }

    enum Status {
        NotStarted, Profiling, Stopped
    };
public:
    template <typename TEvent, typename... TArgs>
    void record_host(TArgs&&... args) {
        MGB_LOCK_GUARD(m_lock);
        if (!m_event_mask.test(index_of<TEvent>())) {
            return;
        }
        mgb_assert(m_status != Stopped, "record after stop");
        auto instant = HostInstant{std::this_thread::get_id(), m_host_timer.get_msecs()};
        m_record_list.emplace_back(EventRecord<TEvent>{std::move(instant), {std::forward<TArgs>(args)...}});
    }
    template <typename TEvent, typename... TArgs>
    void record_device(Device device, TArgs&&... args) {
        MGB_LOCK_GUARD(m_lock);
        if (!m_event_mask.test(index_of<TEvent>())) {
            return;
        }
        mgb_assert(m_status != Stopped, "record after stop");
        auto before = m_host_timer.get_msecs();
        auto event = m_device_timer.get_device_time(device);
        auto after = m_host_timer.get_msecs();
        auto instant = DeviceInstant{before, event, after};
        m_record_list.emplace_back(EventRecord<TEvent>{std::move(instant), {std::forward<TArgs>(args)...}});
    }
    // unsafe
    bool is_profiling() {
        return m_status == Profiling;
    }
    void start(Mask mask) {
        MGB_LOCK_GUARD(m_lock);
        mgb_assert(m_status == NotStarted, "profiler already started");
        m_status = Profiling;
        m_event_mask = mask;
        m_host_timer.reset();
    }
    Data stop() {
        MGB_LOCK_GUARD(m_lock);
        mgb_assert(m_status == Profiling, "profiler not active");
        m_status = Stopped;
        for (auto&& record: m_record_list) {
            std::visit([&](const auto& record){
                record.wait();
            }, record);
        }
        auto records = std::move(m_record_list);
        return { records, m_host_timer.get_started_at() };
    }
protected:
    std::vector<Record> m_record_list;
    Mask m_event_mask;
    std::atomic<Status> m_status = NotStarted;
};

}  // namespace imperative
}  // namespace mgb
