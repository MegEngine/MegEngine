/**
 * \file imperative/src/impl/interpreter/profiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./profiler.h"

#include <sstream>
#include <cinttypes>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#elif defined(_WIN32)
#include <process.h>
#else
#error Unsupported platform
#endif

#include "../op_trait.h"

namespace mgb::imperative::interpreter::intl {

namespace {

struct InterpreterProfilerDumpChromeTimelineContext {
    // either host_thread(std::thread::id) or device_thread(CompNode)
    using Thread = std::variant<std::thread::id, CompNode>;

    // input params
    std::string base_name;
    std::string format;
    InterpreterProfiler::Data profile_data;
    InterpreterProfiler::Option option;
    std::function<std::string(std::thread::id)> host_map;

    // internal states
    decltype(getpid()) pid;
    CompNode::UnorderedMap<std::map<double, CompNode::Event*>> device_sync_map;
    SmallVector<Thread> thread_list;
    double time_start;
    // options
    bool show_operator_name;
    // results
    ChromeTraceEventList event_list;

    InterpreterProfilerDumpChromeTimelineContext(
            std::string base_name,
            std::string format,
            InterpreterProfiler::Data profile_data,
            InterpreterProfiler::Option option,
            std::function<std::string(std::thread::id)> host_map)
        : base_name{base_name}, format{format}, profile_data{profile_data}, option{option}, host_map{host_map} {
        pid = getpid();
        time_start = option.align_time ? time_start : 0;
        show_operator_name = option.show_operator_name;
    }

    // get device time from event
    double get_device_time(CompNode::Event* device_event, double host_time) {
        device_event->host_wait();
        auto& sync_map = device_sync_map[device_event->comp_node()];
        // find sync point
        auto iter = sync_map.begin();
        auto sync_current = [&] {
            iter = sync_map.insert(iter, {host_time, device_event});
            return host_time;
        };
        if (iter == sync_map.end()) {
            // not found, insert sync
            return sync_current();
        }
        auto& [base_time, base] = *iter;
        // calculate elapsed time
        double delta_time = base->elapsed_time_until(*device_event) * 1e3;
        return base_time + delta_time;
    };

    template <typename T>
    size_t get_tid(T t) {
        for (size_t i = 0; i < thread_list.size(); i++) {
            if (thread_list[i] == Thread{t}) {
                return i;
            }
        }
        thread_list.push_back(t);
        return thread_list.size() - 1;
    };

    ChromeTraceEvent& new_event(std::string name, char ph, uint64_t tid, double ts) {
        return event_list.new_event().name(name).ph(ph).tid(tid).ts(ts).pid(pid);
    };

    // convert Command to json object. Has to be an callable object
    static auto constexpr cmd_to_args = [](const auto& cmd) {
        auto args = json::Object::make();
        cmd.get_props([&](const char* key, auto&& value){
            (*args)[key] = json::String::make(to_string(value));
        });
        (*args)["__type__"] = json::String::make(typeid(cmd).name());
        return args;
    };

    void process() {
        // enumerate and process each record
        for (auto& record: profile_data.records) {
            std::visit([this](const auto& record){
                using TEvent = std::decay_t<decltype(record.data)>;
                Session<TEvent>(*this, record).process();
            }, record);
        }
        for (size_t tid = 0; tid < thread_list.size(); ++tid) {
            auto tname = std::visit([&](auto host_or_device) -> std::string{
                using T = std::decay_t<decltype(host_or_device)>;
                if constexpr (std::is_same_v<T, std::thread::id>) {
                    // take name from host_map
                    return host_map(host_or_device);
                } else {
                    // use CompNode::to_string
                    return host_or_device.to_string();
                }
            }, thread_list[tid]);
            // assign thread name
            new_event("thread_name", 'M', tid, 0)
                .arg("name", tname);
        }
        // wraite output to file
        std::string out_buf;
        event_list.to_json()->writeto(out_buf, 4);
        std::ofstream output_stream;
        output_stream.open(base_name + "." + format);
        output_stream << out_buf;
        output_stream.flush();
        output_stream.close();
    }

    template <typename TEvent>
    struct Session {
        InterpreterProfilerDumpChromeTimelineContext& ctx;
        const ProfilerBase::EventRecord<TEvent>& record;
        const TEvent& data;

        Session(InterpreterProfilerDumpChromeTimelineContext& ctx,
                const ProfilerBase::EventRecord<TEvent>& record)
            : ctx{ctx}, record{record}, data{record.data} {}

        uint64_t get_host_tid() {
            return ctx.get_tid(record.host().tid);
        };
        double get_host_ts() {
            return (ctx.time_start + record.host().time) * 1e3;
        };
        uint64_t get_device_tid() {
            return ctx.get_tid(record.device().event->comp_node());
        };
        double get_device_ts() {
            return (ctx.time_start + ctx.get_device_time(record.device().event.get(), record.device().after)) * 1e3;
        };
        ChromeTraceEvent& new_host_event(std::string name, char ph) {
            return ctx.new_event(std::move(name), ph, get_host_tid(), get_host_ts());
        };
        ChromeTraceEvent& new_device_event(std::string name, char ph) {
            return ctx.new_event(std::move(name), ph, get_device_tid(), get_device_ts());
        };

        void process() {
            // dispatch event by type
            if constexpr (std::is_same_v<TEvent, CommandEnqueueEvent>) {
                auto args = std::visit(cmd_to_args, data.icmd.second);
                new_host_event("CommandEnqueue", 'X').dur(0).args(args);
            } else if constexpr (std::is_same_v<TEvent, CommandExecuteEvent>) {
                auto args = std::visit(cmd_to_args, data.icmd.second);
                new_host_event("CommandExecute", 'B').args(args);
            } else if constexpr (std::is_same_v<TEvent, CommandFinishEvent>) {
                new_host_event("CommandExecute", 'E');
            } else if constexpr (std::is_same_v<TEvent, HostOpExecuteEvent>) {
                auto args = json::Object::make();
                auto props = OpDef::props(*data.op);
                auto name = data.op->trait()->name;
                for (auto&& [prop_name, prop_val]: props) {
                    (*args)[std::string("op.") + prop_name] = json::String::make(prop_val);
                }
                (*args)["name"] = json::String::make(name);
                (*args)["id"] = json::Number::make(data.id);
                (*args)["inputs"] = json::String::make(to_string(data.inputs));
                (*args)["outputs"] = json::String::make(to_string(data.outputs));
                new_host_event(ctx.show_operator_name ? name : "OpExecute", 'B').args(args);
            } else if constexpr (std::is_same_v<TEvent, DeviceOpExecuteEvent>) {
                auto args = json::Object::make();
                auto props = OpDef::props(*data.op);
                auto name = data.op->trait()->name;
                for (auto&& [prop_name, prop_val]: props) {
                    (*args)[std::string("op.") + prop_name] = json::String::make(prop_val);
                }
                (*args)["name"] = json::String::make(name);
                (*args)["id"] = json::Number::make(data.id);
                (*args)["inputs"] = json::String::make(to_string(data.inputs));
                (*args)["outputs"] = json::String::make(to_string(data.outputs));
                new_device_event(ctx.show_operator_name ? name : "OpExecute", 'B').args(args);
            } else if constexpr (std::is_same_v<TEvent, HostOpFinishEvent>) {
                auto name = data.op->trait()->name;
                new_host_event(ctx.show_operator_name ? name : "OpExecute", 'E');
            } else if constexpr (std::is_same_v<TEvent, DeviceOpFinishEvent>) {
                auto name = data.op->trait()->name;
                new_device_event(ctx.show_operator_name ? name : "OpExecute", 'E');
            } else if constexpr (std::is_same_v<TEvent, TensorDeclareEvent>) {
                json::Number::make(data.tensor_id);
                new_host_event("TensorLifetime", 'N').id(data.tensor_id);
            } else if constexpr (std::is_same_v<TEvent, TensorProduceEvent>) {
                auto snapshot = json::Object::make();
                (*snapshot)["shape"] = json::String::make(to_string((TensorShape)data.layout));
                (*snapshot)["dtype"] = json::String::make(to_string(data.layout.dtype));
                (*snapshot)["device"] = json::String::make(to_string(data.device));
                json::Number::make(data.tensor_id);
                new_host_event("TensorLifetime", 'O').id(data.tensor_id).arg("snapshot", snapshot);
            } else if constexpr (std::is_same_v<TEvent, TensorEraseEvent>) {
                json::Number::make(data.tensor_id);
                new_host_event("TensorLifetime", 'D').id(data.tensor_id);
            } else if constexpr (std::is_same_v<TEvent, TensorGetPropEvent>) {
                auto args = json::Object::make();
                (*args)["id"] = json::Number::make(data.tensor_id);
                (*args)["prop"] = json::String::make(to_string(data.prop));
                (*args)["prop_desc"] = json::String::make(data.prop_desc);
                new_host_event("TensorGetProp", 'X').dur(0).args(args);
            } else if constexpr (std::is_same_v<TEvent, TensorNotifyPropEvent>) {
                // TODO
            } else if constexpr (std::is_same_v<TEvent, TensorWaitPropEvent>) {
                auto args = json::Object::make();
                (*args)["id"] = json::Number::make(data.tensor_id);
                (*args)["prop"] = json::String::make(to_string(data.prop));
                (*args)["prop_desc"] = json::String::make(data.prop_desc);
                new_host_event("TensorWaitProp", 'B').args(args);
            } else if constexpr (std::is_same_v<TEvent, TensorWaitPropFinishEvent>) {
                auto args = json::Object::make();
                (*args)["id"] = json::Number::make(data.tensor_id);
                (*args)["prop"] = json::String::make(to_string(data.prop));
                (*args)["prop_desc"] = json::String::make(data.prop_desc);
                new_host_event("TensorWaitProp", 'E').args(args);
            } else if constexpr (std::is_same_v<TEvent, SyncStartEvent>) {
                new_host_event("SyncEvent", 'B');
            } else if constexpr (std::is_same_v<TEvent, SyncFinishEvent>) {
                new_host_event("SyncEvent", 'E');
            } else if constexpr (std::is_same_v<TEvent, ChannelBeginScope>) {
                new_host_event(data.name, 'B');
            } else if constexpr (std::is_same_v<TEvent, ChannelEndScope>) {
                new_host_event(data.name, 'E');
            } else if constexpr (std::is_same_v<TEvent, WorkerBeginScope>) {
                new_host_event(data.name, 'B');
            } else if constexpr (std::is_same_v<TEvent, WorkerEndScope>) {
                new_host_event(data.name, 'E');
            } else if constexpr (std::is_same_v<TEvent, DeviceBeginScope>) {
                new_device_event(data.name, 'B');
            } else if constexpr (std::is_same_v<TEvent, DeviceEndScope>) {
                new_device_event(data.name, 'E');
            } else {
                static_assert(!std::is_same_v<TEvent, TEvent>);
            }
        }
    };
};

}

void InterpreterProfiler::dump_data(
        std::string basename,
        std::string format,
        InterpreterProfiler::Data profile_data,
        const InterpreterProfiler::Option& option,
        std::function<std::string(std::thread::id)> host_map) {
    InterpreterProfilerDumpChromeTimelineContext{
        basename, format, profile_data, option, host_map
    }.process();
}

}
