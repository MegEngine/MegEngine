#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
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
    Timer(std::string name, bool default_enabled = true);

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
    struct TimerNode {
        std::map<std::string, std::unique_ptr<TimerNode>> children;
        stats::Timer* timer = nullptr;

        TimerNode() {}
    };

    static inline TimerNode sm_root;

    // register your timers here
    // for example:
    //
    // static inline stats::Timer mytimer;
    //
    // then use MGE_TIMER_SCOPE(mytimer) to collect durations in your code

    static std::pair<long, long> print_node(
            std::string name, TimerNode& node, size_t indent = 0) {
        auto print_indent = [&] {
            for (size_t i = 0; i < indent; ++i) {
                printf(" ");
            }
        };
        long ns = 0, count = 0;
        if (auto* timer = node.timer) {
            print_indent();
            printf("%s costs %'ld ns, hits %'ld times\n", name.c_str(),
                   (long)timer->get().count(), (long)timer->count());
            ns = timer->get().count();
            count = timer->count();
        }
        if (!node.children.empty()) {
            bool collect_children = node.timer == nullptr;
            if (collect_children) {
                print_indent();
                printf("%s:\n", name.c_str());
            }
            long ns = 0, count = 0;
            for (auto&& child : node.children) {
                auto [child_ns, child_count] =
                        print_node(child.first, *child.second, indent + 4);
                if (collect_children) {
                    ns += child_ns;
                    count += child_count;
                }
            }
            if (collect_children) {
                print_indent();
                printf("total costs %'ld ns, hits %'ld times\n", ns, count);
            }
        }
        return {ns, count};
    }

    static void print() {
        for (auto&& child : sm_root.children) {
            print_node(child.first, *child.second);
        }
    }

    static void reset() {
        auto reset_node = [](TimerNode& node, auto&& reset_node) -> void {
            if (auto* timer = node.timer) {
                timer->reset();
            }
            for (auto&& child : node.children) {
                reset_node(*child.second, reset_node);
            }
        };
        reset_node(sm_root, reset_node);
    }
};

inline stats::Timer::Timer(std::string name, bool default_enabled)
        : m_name(name), m_default_enabled(default_enabled) {
    std::vector<std::string> terms;
    Stats::TimerNode* node = &Stats::sm_root;
    while (true) {
        auto pos = name.find(".");
        if (pos == std::string::npos) {
            auto& child = node->children[name];
            child = std::make_unique<Stats::TimerNode>();
            node = child.get();
            node->timer = this;
            break;
        } else {
            auto& child = node->children[name.substr(0, pos)];
            if (!child) {
                child = std::make_unique<Stats::TimerNode>();
            }
            node = child.get();
            name = name.substr(pos + 1);
        }
    }
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
