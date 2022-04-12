#pragma once

#include <chrono>
#include <ctime>

#include "megbrain/common.h"
#include "megbrain/imperative/utils/to_string.h"

namespace mgb::imperative::cupti {

struct clock {
    typedef std::chrono::nanoseconds duration;
    typedef duration::rep rep;
    typedef duration::period period;
    typedef std::chrono::time_point<clock> time_point;
    static const bool is_steady = false;

    static time_point now() /* noexcept */;
};

using time_point = clock::time_point;

using duration = clock::duration;

struct device_t {
    uint32_t device_id;

    bool operator==(const device_t& rhs) const { return device_id == rhs.device_id; }
};

struct context_t : device_t {
    uint32_t context_id;

    bool operator==(const context_t& rhs) const {
        return device_t::operator==(rhs) && context_id == rhs.context_id;
    }
};

struct stream_t : context_t {
    uint32_t stream_id;

    bool operator==(const stream_t& rhs) const {
        return context_t::operator==(rhs) && stream_id == rhs.stream_id;
    }
};

bool available();

void enable();

void disable();

void flush();

bool enabled();

template <typename TActivity>
struct activity {
private:
    TActivity* m_ptr;

public:
    activity(void* ptr) : m_ptr((TActivity*)ptr) {}

    time_point start() const { return time_point(duration(m_ptr->start)); }

    time_point end() const { return time_point(duration(m_ptr->end)); }

    device_t device() const { return {m_ptr->deviceId}; }

    context_t context() const { return {device(), m_ptr->contextId}; }

    stream_t stream() const { return {context(), m_ptr->streamId}; }

    TActivity* operator->() const { return m_ptr; }
};

}  // namespace mgb::imperative::cupti

template <>
class std::hash<mgb::imperative::cupti::stream_t> {
public:
    size_t operator()(const mgb::imperative::cupti::stream_t& value) const {
        return value.stream_id;
    }
};
