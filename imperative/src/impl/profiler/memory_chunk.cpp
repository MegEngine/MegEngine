#include <map>
#include <vector>
#include <array>

#include "megbrain/imperative/utils/to_string.h"
#include "megbrain/utils/debug.h"

#include "./formats.h"
#include "./states.h"

#include "./events.h"

namespace mgb::imperative::profiler {

class XMLWriter {
private:
    std::vector<std::vector<std::string>> elements;
public:
    struct ElementGuard {
        XMLWriter* writer;
        std::string name;
        std::vector<std::pair<std::string, std::string>> attrs;

        template <typename T>
        ElementGuard& attr(std::string key, T&& value) {
            attrs.push_back({key, mgb::imperative::to_string(value)});
            return *this;
        }

        std::string to_string_start() const {
            std::string builder;
            builder.append(ssprintf("<%s",
                        name.c_str()));
            for (auto&& [k, v]: attrs) {
                builder.append(ssprintf(" %s=\"%s\"", k.c_str(), v.c_str()));
            }
            builder.append(">\n");
            return builder;
        }

        std::string to_string_end() const {
            return ssprintf("</%s>\n", name.c_str());
        }

        ElementGuard(XMLWriter* writer, std::string name): writer{writer}, name{name} {
            writer->elements.emplace_back();
        }

        ~ElementGuard() {
            auto children = std::move(writer->elements.back());
            writer->elements.pop_back();
            std::string builder;
            builder.append(to_string_start());
            for (auto&& child: children) {
                builder.append(child);
            }
            builder.append(to_string_end());
            writer->elements.back().push_back(builder);
        }
    };
    XMLWriter() {
        elements.emplace_back().push_back("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    }
    ElementGuard element(std::string tag) {
        return ElementGuard{this, tag};
    }
    void text(std::string text) {
        elements.back().push_back(text);
    }
    void doctype(std::string element, std::string dtd, std::vector<std::string> args) {
        std::string builder = ssprintf("<!DOCTYPE %s %s", element.c_str(), dtd.c_str());
        for (auto&& arg: args) {
            builder.append(ssprintf(" %s", arg.c_str()));
        }
        builder.append(">\n");
        elements.back().push_back(builder);
    }
    std::string to_string() const {
        mgb_assert(elements.size() == 1 && elements[0].size() >= 1);
        std::string builder;
        for (auto&& element: elements[0]) {
            builder.append(element);
        }
        return builder;
    }
};

struct MemoryChunk {
    std::array<uintptr_t, 2> address;
    std::string name;
    TensorLayout layout;
    std::array<profiler::Duration, 2> time;
    std::optional<uint64_t> group;

    bool empty() const {
        return address[1] - address[0] == 0;
    }
};

struct MemoryFlow {
    std::unordered_map<uint64_t, MemoryChunk> chunks;

    std::pair<uintptr_t, uintptr_t> address_range() const {
        auto addr_begin = std::numeric_limits<uintptr_t>::max();
        auto addr_end = std::numeric_limits<uintptr_t>::min();
        for(auto&& [id, chunk]: chunks) {
            MGB_MARK_USED_VAR(id);
            if (chunk.empty()) continue;
            addr_begin = std::min(addr_begin, chunk.address[0]);
            addr_end = std::max(addr_end, chunk.address[1]);
        }
        return {addr_begin, addr_end};
    }

    std::pair<profiler::Duration, profiler::Duration> time_range() const {
        auto time_begin = profiler::Duration::max();
        auto time_end = profiler::Duration::min();
        for(auto&& [id, chunk]: chunks) {
            MGB_MARK_USED_VAR(id);
            if (chunk.empty()) continue;
            time_begin = std::min(time_begin, chunk.time[0]);
            time_end = std::max(time_end, chunk.time[1]);
        }
        return {time_begin, time_end};
    }

    XMLWriter to_svg() const {
        XMLWriter writer;
        auto&& [addr_begin, addr_end] = address_range();
        auto&& [time_begin, time_end] = time_range();
        writer.doctype("svg", "PUBLIC", {
                "\"-//W3C//DTD SVG 1.1//EN\"",
                "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\""
        });
        auto svg = writer.element("svg");
        svg.attr("xmlns", std::string{"http://www.w3.org/2000/svg"});
        svg.attr("xmlns:tag", std::string{"https://megengine.org.cn"});
        double time_scale = 1e5;
        double addr_scale = 1e6;
        svg.attr("width", (time_end-time_begin).count()/time_scale);
        svg.attr("height", (addr_end-addr_begin)/addr_scale);
        {
            auto rect = writer.element("rect");
            rect.attr("x", 0);
            rect.attr("y", 0);
            rect.attr("width", (time_end-time_begin).count()/time_scale);
            rect.attr("height", (addr_end-addr_begin)/addr_scale);
            rect.attr("fill", std::string{"blue"});
        }
        double us = 1e3, ms = 1e6;
        std::map<double, std::string> time2color = {
            {0 * us, "#DDDDDD"},
            {100 * us, "#CCCCCC"},
            {1 * ms, "#BBBBBB"},
            {10 * ms, "#AAAAAA"},
            {100 * ms, "#999999"},
            {1000 * ms, "#888888"},
            {std::numeric_limits<double>::infinity(), "#555555"},
        };
        auto time2str = [](profiler::Duration ns){
            using pair_t = std::pair<uint64_t, const char*>;
            static pair_t units[] = {
                {1, "ns "},
                {1e3, "us "},
                {1e6, "ms "},
                {1e9, "s "},
            };
            std::string builder;
            auto comparator = [](const pair_t& lhs, const pair_t& rhs) {
                return lhs.first < rhs.first;
            };
            while (ns.count() > 0) {
                auto iter = std::upper_bound(std::begin(units), std::end(units), std::make_pair(ns.count(), ""), comparator) - 1;
                builder += std::to_string(ns.count() / iter->first) + iter->second;
                ns = ns % iter->first;
            }
            return builder;
        };
        auto size2str = [](size_t sz){
            using pair_t = std::pair<size_t, const char*>;
            static pair_t units[] = {
                {1, "B "},
                {1024, "KB "},
                {1024*1024, "MB "},
                {1024*1024*1024, "GB "},
            };
            std::string builder;
            auto comparator = [](const pair_t& lhs, const pair_t& rhs) {
                return lhs.first < rhs.first;
            };
            while (sz > 0) {
                auto iter = std::upper_bound(std::begin(units), std::end(units), std::make_pair(sz, ""), comparator) - 1;
                builder += std::to_string(sz / iter->first) + iter->second;
                sz = sz % iter->first;
            }
            return builder;
        };
        for (auto&& [id, chunk]: chunks) {
            MGB_MARK_USED_VAR(id);
            if (chunk.empty()) continue;
            double left = (chunk.time[0]-time_begin).count()/time_scale;
            double right = (chunk.time[1]-time_begin).count()/time_scale;
            double top = (chunk.address[0]-addr_begin)/addr_scale;
            double bottom = (chunk.address[1]-addr_begin)/addr_scale;
            double duration = (chunk.time[1] - chunk.time[0]).count();
            {
                auto rect = writer.element("rect");
                rect.attr("x", left);
                rect.attr("y", top);
                rect.attr("height", bottom - top);
                rect.attr("width", right - left);
                rect.attr("fill", time2color.lower_bound(duration)->second);
                auto mge_attr = [&](const char* name, auto&& value) {
                    rect.attr(ssprintf("tag:%s", name), value);
                };
                mge_attr("type", std::string("tensor"));
                mge_attr("name", chunk.name);
                mge_attr("address", ssprintf("%p", reinterpret_cast<void*>(chunk.address[0])));
                mge_attr("size", size2str(chunk.address[1] - chunk.address[0]));
                mge_attr("layout", chunk.layout.to_string());
                mge_attr("produced", time2str(chunk.time[0]));
                mge_attr("erased", time2str(chunk.time[1]));
                mge_attr("duration", time2str(chunk.time[1] - chunk.time[0]));
                if (chunk.group) {
                    mge_attr("group", std::to_string(*chunk.group));
                }
            }
        }
        return writer;
    }
};

struct MemoryFlowVisitor: EventVisitor<MemoryFlowVisitor> {
    MemoryFlow memory_flow;

    template <typename TEvent>
    void visit_event(const TEvent &event) {
        if constexpr (std::is_same_v<TEvent, TensorProduceEvent>) {
            auto& chunk = memory_flow.chunks[event.tensor_id];
            uint64_t address = reinterpret_cast<uintptr_t>(event.ptr);
            auto span = event.layout.span();
            auto dtype = event.layout.dtype;
            // assume dtype is not lowbit
            if (!address) {
                chunk.address = {0, 0};
            } else {
                chunk.address = {address+span.low_elem*dtype.size(), address+span.high_elem*dtype.size()};
            }
            chunk.layout = event.layout;
            chunk.time[0] = since_start(to_device_time(current->time, current_tensor->device));
            chunk.name = current_tensor->name;
            chunk.group = current_tensor->source;
        } else if constexpr (std::is_same_v<TEvent, TensorReleaseEvent>) {
            auto& chunk = memory_flow.chunks[event.tensor_id];
            chunk.time[1] = since_start(to_device_time(current->time, current_tensor->device));
        }
    }

    void notify_counter(std::string key, int64_t old_val, int64_t new_val) {}
};

void dump_memory_flow(std::string filename, Profiler::bundle_t result) {
    MemoryFlowVisitor visitor;
    visitor.process_events(std::move(result));
    debug::write_to_file(filename.c_str(), visitor.memory_flow.to_svg().to_string());
}

}
