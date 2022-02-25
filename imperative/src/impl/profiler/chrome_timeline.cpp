#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#elif defined(_WIN32)
#include <process.h>
#else
#error Unsupported platform
#endif

#include "nlohmann/json.hpp"

#include "megbrain/imperative/utils/platform.h"
#include "megbrain/utils/debug.h"

#include "./formats.h"
#include "./states.h"

namespace mgb::imperative::profiler {

class ChromeTraceEvent {
public:
    ChromeTraceEvent& name(std::string name) {
        m_name = std::move(name);
        return *this;
    }
    ChromeTraceEvent& tid(uint64_t tid) {
        m_tid = std::move(tid);
        return *this;
    }
    ChromeTraceEvent& cat(std::string cat) {
        m_cat = std::move(cat);
        return *this;
    }
    ChromeTraceEvent& scope(std::string scope) {
        m_scope = std::move(scope);
        return *this;
    }
    ChromeTraceEvent& pid(uint64_t pid) {
        m_pid = pid;
        return *this;
    }
    ChromeTraceEvent& id(uint64_t id) {
        m_id = id;
        return *this;
    }
    ChromeTraceEvent& idx(uint64_t idx) {
        m_idx = idx;
        return *this;
    }
    template <typename TDuration>
    ChromeTraceEvent& ts(TDuration ts) {
        m_ts = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(ts)
                       .count();
        return *this;
    }
    ChromeTraceEvent& dur(uint64_t dur) {
        m_dur = dur;
        return *this;
    }
    ChromeTraceEvent& ph(char ph) {
        m_ph = ph;
        return *this;
    }
    ChromeTraceEvent& bp(char bp) {
        m_bp = bp;
        return *this;
    }
    ChromeTraceEvent& args(nlohmann::json args) {
        m_args = std::move(args);
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, std::string value) {
        m_args[key] = value;
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, double value) {
        m_args[key] = value;
        return *this;
    }
    ChromeTraceEvent& stack(Trace trace) {
        m_stack = std::move(trace);
        return *this;
    }

    nlohmann::json to_json() const {
        nlohmann::json result;
        auto prop_str = [&](auto key, auto value) {
            if (value.empty()) {
                return;
            }
            result[key] = value;
        };
        auto prop_num = [&](auto key, auto value) {
            if (!value) {
                return;
            }
            result[key] = value.value();
        };
        auto prop_char = [&](auto key, auto value) {
            if (!value) {
                return;
            }
            result[key] = std::string{} + value.value();
        };
        prop_str("name", m_name);
        prop_str("cat", m_cat);
        prop_str("scope", m_scope);
        prop_num("tid", m_tid);
        prop_num("pid", m_pid);
        prop_num("id", m_id);
        prop_num("idx", m_idx);
        prop_num("ts", m_ts);
        prop_num("dur", m_dur);
        prop_char("ph", m_ph);
        prop_char("bp", m_bp);
        if (!m_args.empty()) {
            result["args"] = m_args;
        }
        if (m_stack) {
            nlohmann::json stack;
            for (auto&& frame : m_stack->frames()) {
                stack.push_back(
                        ssprintf("%s%ld", frame.node->name().c_str(), frame.version));
            }
            std::reverse(stack.begin(), stack.end());
            result["stack"] = stack;
        }
        return result;
    }

private:
    std::string m_name;
    std::string m_cat;
    std::string m_scope;

    std::optional<uint64_t> m_tid;
    std::optional<uint64_t> m_pid;
    std::optional<uint64_t> m_id;
    std::optional<uint64_t> m_idx;
    std::optional<double> m_ts;
    std::optional<uint64_t> m_dur;
    std::optional<char> m_ph;
    std::optional<char> m_bp;
    nlohmann::json m_args;
    std::optional<Trace> m_stack;
};

class ChromeTraceEvents {
public:
    ChromeTraceEvent& new_event() {
        m_content.emplace_back();
        return m_content.back();
    }

    std::string& metadata(std::string key) { return m_metadata[key]; }

    nlohmann::json to_json() const {
        nlohmann::json result;
        nlohmann::json event_list;
        nlohmann::json metadata;
        for (auto&& event : m_content) {
            event_list.push_back(event.to_json());
        }
        for (auto&& [key, value] : m_metadata) {
            metadata[key] = value;
        }
        result["traceEvents"] = event_list;
        result["metadata"] = metadata;
        return result;
    }

    std::string to_string() const {
        auto json = to_json();
        return "{"
               "\"traceEvents\":" +
               nlohmann::to_string(json["traceEvents"]) +
               ","
               "\"metadata\":" +
               nlohmann::to_string(json["metadata"]) + "}";
    }

private:
    std::vector<ChromeTraceEvent> m_content;
    std::unordered_map<std::string, std::string> m_metadata;
};

struct ChromeTimelineEventVisitor : EventVisitor<ChromeTimelineEventVisitor> {
    ChromeTraceEvents trace_events;
    decltype(getpid()) pid = getpid();
    std::string pid_str = std::to_string(pid);

    ChromeTimelineEventVisitor() {}

    ChromeTraceEvent& new_event(
            std::string name, char ph, size_t tid, profiler::HostTime time) {
        return trace_events.new_event().name(name).ph(ph).pid(pid).tid(tid).ts(
                since_start(time));
    }

    ChromeTraceEvent& new_host_event(std::string name, char ph) {
        return trace_events.new_event()
                .name(name)
                .ph(ph)
                .pid(pid)
                .tid(to_tid(current->tid))
                .ts(since_start(current->time));
    }

    ChromeTraceEvent& new_cupti_event(
            std::string name, char ph, cupti::stream_t stream,
            cupti::time_point timestamp) {
        return new_event(name, ph, to_tid(stream), time_from_cupti(timestamp));
    }

    ChromeTraceEvent& new_device_event(std::string name, char ph, CompNode device) {
        auto time = since_start(to_device_time(current->time, device));
        return trace_events.new_event()
                .name(name)
                .ph(ph)
                .pid(pid)
                .tid(to_tid(device))
                .ts(time);
    }

    const char* to_cstr(TensorCommandKind kind) {
        switch (kind) {
            case TensorCommandKind::Put:
                return "Put";
            case TensorCommandKind::Drop:
                return "Drop";
            case TensorCommandKind::Del:
                return "Del";
            case TensorCommandKind::RecFree:
                return "RecFree";
            case TensorCommandKind::ReGen:
                return "ReGen";
            case TensorCommandKind::GetValue:
                return "GetValue";
        }
        return "UnknownCommand";
    }

    template <typename TEvent>
    void visit_event(const TEvent& event) {
        if constexpr (std::is_same_v<TEvent, OpDispatchEvent>) {
            new_host_event("OpDispatch", 'B');
            new_host_event(pid_str, 's')
                    .cat("OpDispatch")
                    .id(event.op_id)
                    .scope(pid_str);
            new_host_event("OpDispatch", 'E').args(current_op->detail());
        } else if constexpr (std::is_same_v<TEvent, OpExecuteEvent>) {
            mgb_assert(event.op_id != 0);
            new_host_event(current_op->name, 'B');
            new_host_event(pid_str, 't')
                    .cat("OpDispatch")
                    .id(current_op->id)
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, OpExecuteFinishEvent>) {
            new_host_event(current_op->name, 'E').args(current_op->detail());
        } else if constexpr (std::is_same_v<TEvent, KernelLaunchEvent>) {
            new_host_event(pid_str, 's')
                    .id(event.kernel_id)
                    .cat("KernelLaunch")
                    .scope(pid_str);
            new_device_event(current_op->name, 'B', event.device).cat("Kernel");
            new_device_event(pid_str, 'f', event.device)
                    .id(event.kernel_id)
                    .bp('e')
                    .cat("KernelLaunch")
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, KernelLaunchFinishEvent>) {
            new_device_event(current_op->name, 'E', event.device)
                    .cat("Kernel")
                    .args(current_op->detail());
        } else if constexpr (std::is_same_v<TEvent, TensorProduceEvent>) {
            if (current_tensor->living_time == profiler::Duration::zero()) {
                new_host_event(pid_str, 's')
                        .id(event.tensor_id)
                        .cat("TensorLink")
                        .scope(pid_str);
            } else {
                new_host_event(pid_str, 't')
                        .id(event.tensor_id)
                        .cat("TensorLink")
                        .scope(pid_str);
            }
        } else if constexpr (std::is_same_v<TEvent, TensorUsageEvent>) {
            new_host_event(pid_str, 't')
                    .id(event.tensor_id)
                    .cat("TensorLink")
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, TensorReleaseEvent>) {
            current_tensor->living_time += current->time - current_tensor->produced;
            current_tensor->produced = {};
            new_host_event(pid_str, 't')
                    .id(event.tensor_id)
                    .cat("TensorLink")
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, TensorEraseEvent>) {
            if (current_tensor->living_time != profiler::Duration::zero()) {
                new_host_event(pid_str, 'f')
                        .id(event.tensor_id)
                        .bp('e')
                        .cat("TensorLink")
                        .scope(pid_str);
            }
        } else if constexpr (std::is_same_v<TEvent, TensorGetPropEvent>) {
            new_host_event("TensorGetProp", 'X')
                    .dur(0)
                    .args(current_tensor->detail(current->time))
                    .arg("kind", imperative::to_string(event.prop));
        } else if constexpr (std::is_same_v<TEvent, TensorWaitPropEvent>) {
            new_host_event("TensorWaitProp", 'B');
        } else if constexpr (std::is_same_v<TEvent, TensorWaitPropFinishEvent>) {
            new_host_event(pid_str, 'f')
                    .id(event.tensor_id)
                    .bp('e')
                    .cat("TensorProp")
                    .scope(pid_str);
            new_host_event("TensorWaitProp", 'E')
                    .args(current_tensor->detail(current->time));
        } else if constexpr (std::is_same_v<TEvent, TensorNotifyPropEvent>) {
            new_host_event(pid_str, 's')
                    .id(event.tensor_id)
                    .cat("TensorProp")
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, SampleDeviceFinishEvent>) {
            std::string device_name = event.device.locator().to_string();
            new_host_event(ssprintf("%s_alloc_mem", device_name.c_str()), 'C')
                    .arg("value", event.total_memory - event.free_memory);
        } else if constexpr (std::is_same_v<TEvent, TensorCommandEvent>) {
            new_host_event(
                    ssprintf("%s %zu", to_cstr(event.kind), event.tensor_id), 'B');
        } else if constexpr (std::is_same_v<TEvent, TensorCommandFinishEvent>) {
            new_host_event(
                    ssprintf("%s %zu", to_cstr(event.kind), event.tensor_id), 'E')
                    .args(current_tensor->detail(current->time));
        } else if constexpr (std::is_same_v<TEvent, ScopeEvent>) {
            new_host_event(event.name, 'B');
        } else if constexpr (std::is_same_v<TEvent, ScopeFinishEvent>) {
            new_host_event(event.name, 'E');
        } else if constexpr (std::is_same_v<TEvent, OpInputEvent>) {
            new_host_event(ssprintf("Input %zu", event.tensor_id), 'B')
                    .args(current_tensor->detail(current->time));
        } else if constexpr (std::is_same_v<TEvent, OpInputFinishEvent>) {
            new_host_event(ssprintf("Input %zu", event.tensor_id), 'E')
                    .args(current_tensor->detail(current->time));
        } else if constexpr (std::is_same_v<TEvent, OpOutputEvent>) {
            new_host_event(ssprintf("Output %zu", event.tensor_id), 'B')
                    .args(current_tensor->detail(current->time));
        } else if constexpr (std::is_same_v<TEvent, OpOutputFinishEvent>) {
            new_host_event(ssprintf("Output %zu", event.tensor_id), 'E')
                    .args(current_tensor->detail(current->time));
        } else if constexpr (std::is_same_v<TEvent, StartProfileEvent>) {
            new_host_event("StartProfile", 'B');
        } else if constexpr (std::is_same_v<TEvent, StartProfileFinishEvent>) {
            new_host_event("StartProfile", 'E');
        } else if constexpr (std::is_same_v<TEvent, StopProfileEvent>) {
            new_host_event("StopProfile", 'B');
        } else if constexpr (std::is_same_v<TEvent, StopProfileFinishEvent>) {
            new_host_event("StopProfile", 'E');
        } else if constexpr (std::is_same_v<TEvent, CustomEvent>) {
            new_host_event(event.title, 'B');
            if (event.device.valid()) {
                new_device_event(event.title, 'B', event.device);
            }
        } else if constexpr (std::is_same_v<TEvent, CustomFinishEvent>) {
            new_host_event(event.title, 'E');
            if (event.device.valid()) {
                new_device_event(event.title, 'E', event.device);
            }
        } else if constexpr (std::is_same_v<TEvent, AutoEvictEvent>) {
            new_host_event("AutoEvict", 'B');
        } else if constexpr (std::is_same_v<TEvent, AutoEvictFinishEvent>) {
            new_host_event("AutoEvict", 'E');
        } else if constexpr (std::is_same_v<TEvent, HostToDeviceEvent>) {
            new_device_event("HostToDevice", 'B', event.device);
        } else if constexpr (std::is_same_v<TEvent, HostToDeviceFinishEvent>) {
            new_device_event("HostToDevice", 'E', event.device)
                    .arg("shape", event.layout.TensorShape::to_string())
                    .arg("dtype", event.layout.dtype.name())
                    .arg("nr_elements", event.layout.total_nr_elems())
                    .arg("device", event.device.to_string());
        } else if constexpr (std::is_same_v<TEvent, RecordDeviceEvent>) {
            auto current_host_time = current->time;
            auto current_device_time =
                    to_device_time(current->time, event.event->comp_node());
            auto device_ahead = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_device_time - current_host_time);
            new_host_event("device_ahead_ms", 'C').arg("value", device_ahead.count());
        } else if constexpr (std::is_same_v<TEvent, CUPTIKernelLaunchEvent>) {
            new_host_event(demangle(event.name), 'B');
            new_host_event(pid_str, 's')
                    .id(event.correlation_id)
                    .cat("KernelLink")
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, CUPTIKernelLaunchFinishEvent>) {
            new_host_event(demangle(event.name), 'E');
        } else if constexpr (std::is_same_v<TEvent, CUPTIKernelExecuteEvent>) {
            new_cupti_event(demangle(event.name), 'B', event.stream, event.start)
                    .arg("execution_time", (event.end - event.start).count());
            new_cupti_event(pid_str, 'f', event.stream, event.end)
                    .id(event.correlation_id)
                    .bp('e')
                    .cat("KernelLink")
                    .scope(pid_str);
            new_cupti_event(demangle(event.name), 'E', event.stream, event.end)
                    .arg("execution_time", (event.end - event.start).count());
        } else if constexpr (std::is_same_v<TEvent, CUPTIMemcpyLaunchEvent>) {
            new_host_event("Memcpy", 'B');
            new_host_event(pid_str, 's')
                    .id(event.correlation_id)
                    .cat("CUPTILink")
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, CUPTIMemcpyLaunchFinishEvent>) {
            new_host_event("Memcpy", 'E');
        } else if constexpr (std::is_same_v<TEvent, CUPTIMemcpyEvent>) {
            auto memkind2str = [](uint8_t kind) {
                const char* const valid_kinds[] = {
                        "CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN",
                        "CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE",
                        "CUPTI_ACTIVITY_MEMORY_KIND_PINNED",
                        "CUPTI_ACTIVITY_MEMORY_KIND_DEVICE",
                        "CUPTI_ACTIVITY_MEMORY_KIND_ARRAY",
                        "CUPTI_ACTIVITY_MEMORY_KIND_MANAGED",
                        "CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC",
                        "CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC"};
                if (kind > (sizeof(valid_kinds) / sizeof(const char*))) {
                    return "invalid";
                }
                return valid_kinds[kind];
            };
            new_cupti_event("Memcpy", 'B', event.stream, event.start)
                    .arg("bytes", imperative::to_string(event.bytes))
                    .arg("src_kind", memkind2str(event.src_kind))
                    .arg("dst_kind", memkind2str(event.dst_kind));
            new_cupti_event(pid_str, 'f', event.stream, event.start)
                    .id(event.correlation_id)
                    .bp('e')
                    .cat("CUPTILink")
                    .scope(pid_str);
            new_cupti_event("Memcpy", 'E', event.stream, event.end)
                    .arg("bytes", imperative::to_string(event.bytes))
                    .arg("src_kind", memkind2str(event.src_kind))
                    .arg("dst_kind", memkind2str(event.dst_kind));
        } else if constexpr (std::is_same_v<TEvent, CUPTIMemsetEvent>) {
            new_cupti_event("Memset", 'B', event.stream, event.start)
                    .arg("value", imperative::to_string(event.value))
                    .arg("bytes", imperative::to_string(event.bytes));
            new_cupti_event("Memset", 'E', event.stream, event.start)
                    .arg("value", imperative::to_string(event.value))
                    .arg("bytes", imperative::to_string(event.bytes));
        } else if constexpr (std::is_same_v<TEvent, CUPTIRuntimeEvent>) {
            new_host_event(event.name, 'B');
        } else if constexpr (std::is_same_v<TEvent, CUPTIRuntimeFinishEvent>) {
            new_host_event(event.name, 'E');
        } else if constexpr (std::is_same_v<TEvent, CUPTIDriverEvent>) {
            new_host_event(event.name, 'B');
            new_host_event(pid_str, 's')
                    .id(event.correlation_id)
                    .cat("CUPTILink")
                    .scope(pid_str);
        } else if constexpr (std::is_same_v<TEvent, CUPTIDriverFinishEvent>) {
            new_host_event(event.name, 'E');
        }
    }

    void notify_counter(std::string key, int64_t old_val, int64_t new_val) {
        new_host_event(key, 'C').arg("value", new_val);
    }

    void name_threads(Profiler::thread_dict_t thread_dict) {
        for (auto&& host : host_threads()) {
            if (thread_dict.count(host)) {
                trace_events.new_event()
                        .name("thread_name")
                        .ph('M')
                        .pid(pid)
                        .tid(to_tid(host))
                        .arg("name", thread_dict.at(host));
            }
        }
        for (auto&& device : devices()) {
            trace_events.new_event()
                    .name("thread_name")
                    .ph('M')
                    .pid(pid)
                    .tid(to_tid(device))
                    .arg("name", device.to_string_logical());
        }
    }
};

void dump_chrome_timeline(std::string filename, Profiler::bundle_t result) {
    ChromeTimelineEventVisitor visitor{};
    visitor.process_events(result);
    visitor.name_threads(result.thread_dict);
    auto trace_events = std::move(visitor.trace_events);
    trace_events.metadata("localTime") =
            std::to_string(result.start_at.time_since_epoch().count());
    std::string json_repr = trace_events.to_string();
    mgb::debug::write_to_file(filename.c_str(), json_repr);
}

}  // namespace mgb::imperative::profiler
