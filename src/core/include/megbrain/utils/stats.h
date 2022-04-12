#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "megbrain/common.h"

namespace mgb {
namespace stats {

#define MGE_ENABLE_STATS 1

class Timer {
public:
    using clock_t = std::chrono::system_clock;

private:
    clock_t::duration m_duration = clock_t::duration{0};
    size_t m_timing = 0;
    std::string m_name;
    uint64_t m_count = 0;
    size_t m_enabled = 1;
    bool m_default_enabled = true;

    struct TimeScopeRecursive {
        Timer& timer;
        clock_t::time_point start;
        bool released = false;

        TimeScopeRecursive(Timer& timer) : timer(timer) {
            if (timer.m_enabled && !timer.m_timing++) {
                start = clock_t::now();
            }
        }

        ~TimeScopeRecursive() { release(); }

        void release() {
            if (released) {
                return;
            }
            if (timer.m_enabled) {
                if (!--timer.m_timing) {
                    auto duration = (clock_t::now() - start);
                    timer.m_duration += duration;
                }
                timer.m_count++;
            }
            released = true;
        }
    };

    struct EnableScope {
        Timer& timer;
        bool released = false;

        EnableScope(Timer& timer) : timer(timer) { timer.m_enabled++; }

        ~EnableScope() { release(); }

        void release() {
            if (released) {
                return;
            }
            timer.m_enabled--;
            released = true;
        }
    };

public:
    Timer(std::string name, bool default_enabled = true)
            : m_name(name), m_default_enabled(default_enabled){};

    std::string name() { return m_name; }
    auto time_scope_recursive() { return TimeScopeRecursive(*this); };
    auto enable_scope() { return EnableScope(*this); }
    void reset() {
        m_duration = clock_t::duration{0};
        m_count = 0;
        m_enabled = m_default_enabled ? 1 : 0;
    }

    clock_t::duration get() const { return m_duration; }
    uint64_t count() const { return m_count; }
};
}  // namespace stats

struct Stats {
private:
    struct TimerNode {
        std::map<std::string, std::unique_ptr<TimerNode>> children;
        std::unique_ptr<stats::Timer> timer;
    };

    static TimerNode sm_root;

    // don't register your timers here
    // use MGE_TIMER_SCOPE(mytimer) to collect durations in your code
public:
    MGE_WIN_DECLSPEC_FUC static stats::Timer& get_timer(std::string name);

    MGE_WIN_DECLSPEC_FUC static std::pair<long, long> print_node(
            std::string name, TimerNode& node, size_t indent = 0);

    MGE_WIN_DECLSPEC_FUC static void print();

    MGE_WIN_DECLSPEC_FUC static void reset();
};

#if MGE_ENABLE_STATS

#define MGE_TIMER_SCOPE(name)                                  \
    static auto& _timer_##name = mgb::Stats::get_timer(#name); \
    auto name = _timer_##name.time_scope_recursive()

#define MGE_TIMER_SCOPE_RELEASE(name) name.release()

#define MGE_TIMER_SCOPE_ENABLE(name)                           \
    static auto& _timer_##name = mgb::Stats::get_timer(#name); \
    auto name = _timer_##name.enable_scope()

#else

#define MGE_TIMER_SCOPE(name)         (void)0
#define MGE_TIMER_SCOPE_RELEASE(name) (void)0
#define MGE_TIMER_SCOPE_ENABLE(name)  (void)0

#endif

}  // namespace mgb
