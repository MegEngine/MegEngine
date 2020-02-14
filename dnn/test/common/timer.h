/**
 * \file dnn/test/common/timer.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "test/common/utils.h"

#include <chrono>

namespace megdnn {
namespace test {

class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

public:
    Timer() { reset(); }
    void reset() {
        m_started = false;
        m_stopped = false;
    }
    void start() {
        megdnn_assert(!m_started);
        megdnn_assert(!m_stopped);
        m_started = true;
        m_start_point = clock::now();
    }
    void stop() {
        megdnn_assert(m_started);
        megdnn_assert(!m_stopped);
        m_stopped = true;
        m_stop_point = clock::now();
    }
    size_t get_time_in_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                       m_stop_point - m_start_point)
                .count();
    }

private:
    bool m_started, m_stopped;
    time_point m_start_point, m_stop_point;
};

class Timer2 {
    std::chrono::high_resolution_clock::time_point m_start;

public:
    Timer2() { reset(); }

    void reset() { m_start = std::chrono::high_resolution_clock::now(); }

    double get_secs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now -
                                                                    m_start)
                       .count() *
               1e-9;
    }

    double get_msecs() const { return get_secs() * 1e3; }

    double get_secs_reset() {
        auto ret = get_secs();
        reset();
        return ret;
    }

    double get_msecs_reset() { return get_secs_reset() * 1e3; }
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
