#include "megbrain/utils/stats.h"

namespace mgb {

Stats::TimerNode Stats::sm_root;

stats::Timer& Stats::get_timer(std::string name) {
    auto full_name = name;
    Stats::TimerNode* node = &Stats::sm_root;
    while (true) {
        auto pos = name.find("_");
        if (pos == std::string::npos) {
            auto& child = node->children[name];
            child = std::make_unique<Stats::TimerNode>();
            node = child.get();
            auto& timer = node->timer;
            if (!timer) {
                timer = std::make_unique<stats::Timer>(full_name);
            }
            return *timer;
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

std::pair<long, long> Stats::print_node(
        std::string name, TimerNode& node, size_t indent) {
    auto print_indent = [&] {
        for (size_t i = 0; i < indent; ++i) {
            printf(" ");
        }
    };
    long ns = 0, count = 0;
    if (auto& timer = node.timer) {
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
            auto&& child_res = print_node(child.first, *child.second, indent + 4);
            auto&& child_ns = child_res.first;
            auto&& child_count = child_res.second;
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

void Stats::print() {
    for (auto&& child : sm_root.children) {
        print_node(child.first, *child.second);
    }
}

void Stats::reset() {
    auto reset_node = [](TimerNode& node, auto&& reset_node) -> void {
        if (auto& timer = node.timer) {
            timer->reset();
        }
        for (auto&& child : node.children) {
            reset_node(*child.second, reset_node);
        }
    };
    reset_node(sm_root, reset_node);
}

}  // namespace mgb