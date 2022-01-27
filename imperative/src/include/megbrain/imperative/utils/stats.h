#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace mgb {
namespace imperative {
namespace stats {

#define MGE_ENABLE_STATS 0

class Timer {
public:
    using clock_t = std::chrono::system_clock;

private:
    clock_t::duration m_duration = clock_t::duration{0};
    size_t m_timing = 0;
    const char* m_name = nullptr;
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
                    timer.m_duration += (clock_t::now() - start);
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

    using TimeScope = TimeScopeRecursive;

public:
    Timer(const char* name, bool default_enabled);

    const char* name() { return m_name; }
    auto time_scope() { return TimeScope(*this); }
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
    static inline std::vector<stats::Timer*> sm_timers;

    // register your timers here
    // for example:
    //
    // static inline stats::Timer mytimer;
    //
    // then use MGE_TIMER_SCOPE(mytimer) to collect durations in your code

    static void print() {
        std::vector<const char*> unused_timers;

        for (auto* timer : sm_timers) {
            if (timer->count() == 0) {
                unused_timers.push_back(timer->name());
            } else {
                printf("%s costs %ld ns, happens %ld times\n", timer->name(),
                       timer->get().count(), timer->count());
            }
        }

        if (!unused_timers.empty()) {
            printf("%zu timers unused\n", unused_timers.size());
        }
    }

    static void reset() {
        for (auto* timer : sm_timers) {
            timer->reset();
        }
    }
};

inline stats::Timer::Timer(const char* name, bool default_enabled)
        : m_name(name), m_default_enabled(default_enabled) {
    Stats::sm_timers.push_back(this);
}

#if MGE_ENABLE_STATS
#define MGE_TIMER_SCOPE(name)         auto name = Stats::name.time_scope()
#define MGE_TIMER_SCOPE_RELEASE(name) name.release()
#define MGE_TIMER_SCOPE_ENABLE(name)  auto name = Stats::name.enable_scope()
#else
#define MGE_TIMER_SCOPE(name)         (void)0
#define MGE_TIMER_SCOPE_RELEASE(name) (void)0
#define MGE_TIMER_SCOPE_ENABLE(name)  (void)0
#endif

}  // namespace imperative
}  // namespace mgb
