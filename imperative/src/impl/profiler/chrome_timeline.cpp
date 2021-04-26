/**
 * \file imperative/src/impl/profiler/chrome_timeline.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#elif defined(_WIN32)
#include <process.h>
#else
#error Unsupported platform
#endif

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
    ChromeTraceEvent& ts(uint64_t ts) {
        m_ts = ts;
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
    ChromeTraceEvent& args(std::shared_ptr<json::Object> args) {
        m_args = std::move(args);
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, std::string value) {
        if (!m_args) {
            m_args = json::Object::make();
        }
        (*m_args)[key] = json::String::make(value);
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, double value) {
        if (!m_args) {
            m_args = json::Object::make();
        }
        (*m_args)[key] = json::Number::make(value);
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, std::shared_ptr<json::Value> value) {
        if (!m_args) {
            m_args = json::Object::make();
        }
        (*m_args)[key] = value;
        return *this;
    }

    std::shared_ptr<json::Object> to_json() const {
        auto result = json::Object::make();
        auto prop_str = [&](auto key, auto value) {
            if (value.empty()) {
                return;
            }
            (*result)[key] = json::String::make(value);
        };
        auto prop_num = [&](auto key, auto value) {
            if (!value) {
                return;
            }
            (*result)[key] = json::Number::make(value.value());
        };
        auto prop_char = [&](auto key, auto value) {
            if (!value) {
                return;
            }
            (*result)[key] = json::String::make(std::string{} + value.value());
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
        if (m_args) {
            (*result)["args"] = m_args;
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
    std::optional<uint64_t> m_ts;
    std::optional<uint64_t> m_dur;
    std::optional<char> m_ph;
    std::optional<char> m_bp;
    std::shared_ptr<json::Object> m_args;
};

class ChromeTraceEvents {
public:
    ChromeTraceEvent& new_event() {
        m_content.emplace_back();
        return m_content.back();
    }

    std::shared_ptr<json::Value> to_json() const {
        auto result = json::Object::make();
        auto event_list = json::Array::make();
        for (auto&& event: m_content) {
            event_list->add(event.to_json());
        }
        (*result)["traceEvents"] = event_list;
        return result;
    }
private:
    std::vector<ChromeTraceEvent> m_content;
};


void dump_chrome_timeline(std::string filename, Profiler::options_t options, Profiler::thread_dict_t thread_dict, Profiler::results_t results){
    auto pid = getpid();

    ProfileDataCollector collector;
    ProfileState state;
#define HANDLE_EVENT(type, ...) \
    collector.handle<type>([&](uint64_t id, std::thread::id tid, uint64_t time, type event) __VA_ARGS__ );

    ChromeTraceEvents trace_events;

    #define NEW_HOST(NAME, PH) trace_events.new_event().name(NAME).pid(pid).tid(state[tid].index).ph(PH).ts((double)time/1e3)

    #define NEW_DEVICE(NAME, PH) trace_events.new_event().name(NAME).pid(pid).tid(256+state[event.event->comp_node()].index).ph(PH).ts((double)get_device_time(event.event, time)/1e3)

    #define OP_NAME op_state.name

    #define OP_KERNEL_NAME (op_state.name + "")

    #define OP_PROPS get_op_args(op_state)

    #define OP_ID event.op_id

    #define TENSOR_PROPS get_tensor_args(tensor_state, time)

    #define TENSOR_INFO get_tensor_info(tensor_state, time)

    #define TENSOR_COMMAND_KIND print_tensor_command_kind(event.kind)

    #define HANDLE_PLAIN_EVENT(START, FINISH, NAME_EXPR)\
                HANDLE_EVENT(START, { NEW_HOST(NAME_EXPR, 'B'); })\
                HANDLE_EVENT(FINISH, { NEW_HOST(NAME_EXPR, 'E'); })

    #define HANDLE_TENSOR_EVENT(START, FINISH, NAME_EXPR)\
                HANDLE_EVENT(START, { NEW_HOST(NAME_EXPR, 'B'); })\
                HANDLE_EVENT(FINISH, { auto& tensor_state = state.tensors[event.tensor_id]; NEW_HOST(NAME_EXPR, 'E').args(TENSOR_PROPS); })

    #define INC_COUNTER(NAME, DELTA)\
                { state.statics.NAME += DELTA; NEW_HOST(#NAME, 'C').arg(#NAME, state.statics.NAME); }

    auto get_tensor_args = [](const ProfileTensorState& tensor, uint64_t time) -> std::shared_ptr<json::Object> {
        auto args = json::Object::make();
        (*args)["id"] = json::Number::make(tensor.id);
        (*args)["name"] = json::String::make(tensor.name);
        (*args)["shape"] = json::String::make(tensor.layout.TensorShape::to_string());
        (*args)["dtype"] = json::String::make(tensor.layout.dtype.name());
        (*args)["nr_elements"] = json::Number::make(tensor.layout.total_nr_elems());
        (*args)["device"] = json::String::make(tensor.device.to_string());
        if (tensor.produced) {
            (*args)["living_time"] = json::String::make(std::to_string((time - tensor.produced + tensor.living_time)/1e6) + "ms");
        }
        return args;
    };

    auto get_tensor_info = [](const ProfileTensorState& tensor, uint64_t time) -> std::string {
        std::string name = tensor.name;
        std::string shape = tensor.layout.TensorShape::to_string();
        std::string size_in_bytes = std::to_string(tensor.size_in_bytes());
        std::string device = tensor.device.to_string();
        std::string dtype = tensor.layout.dtype.name();
        return ssprintf("%s(%s:%s:%s)", name.c_str(), shape.c_str(), dtype.c_str(), device.c_str());
    };

    auto get_op_args = [&](const ProfileOperatorState& op) -> std::shared_ptr<json::Object> {
        auto args = json::Object::make();
        auto params = op.params;
        for (auto&& [name, value]: params) {
            (*args)[name] = json::String::make(value);
        }
        (*args)["__id__"] = json::Number::make(op.id);
        (*args)["__name__"] = json::String::make(op.name);
        (*args)["__device__"] = json::String::make(op.device.to_string());
        return args;
    };

    auto get_device_time = [&](const std::shared_ptr<CompNode::Event>& event, uint64_t host) -> uint64_t {
        event->host_wait();
        auto& device_state = state.devices[event->comp_node()];
        if (!device_state.base_event) {
            device_state.base_event = event;
            device_state.base_time = host;
            return host;
        }
        uint64_t device = device_state.base_event->elapsed_time_until(*event) * 1e9 + device_state.base_time;
        return std::max(device, host);
    };

    auto print_tensor_command_kind = [&](int kind) -> const char* {
        switch(kind) {
            case TensorCommandEvent::Put:
                return "Put";
            case TensorCommandEvent::Drop:
                return "Drop";
            case TensorCommandEvent::Del:
                return "Del";
            case TensorCommandEvent::SwapIn:
                return "SwapIn";
            case TensorCommandEvent::SwapOut:
                return "SwapOut";
            case TensorCommandEvent::RecFree:
                return "RecFree";
            case TensorCommandEvent::ReGen:
                return "ReGen";
        }
        return "UnknownCommand";
    };

    HANDLE_EVENT(OpDispatchEvent, {
        auto& op_state = state.operators[OP_ID] = {};
        op_state.id = OP_ID;
        op_state.name = event.op_name;
        op_state.params = event.op_params();
        op_state.inputs = event.inputs;
        op_state.outputs = event.outputs;
        NEW_HOST("OpDispatch", 'B');
        NEW_HOST(ssprintf("%d", pid), 's')
                .cat("OpDispatch")
                .id(OP_ID)
                .scope(std::to_string(pid));
        NEW_HOST("OpDispatch", 'E').args(OP_PROPS);
        INC_COUNTER(op_enqueue_count, 1);
    });

    HANDLE_EVENT(OpExecuteEvent, {
        mgb_assert(OP_ID != 0);
        mgb_assert(state.operators.count(OP_ID) > 0);
        auto& op_state = state.operators[OP_ID];
        op_state.host_begin = time;
        NEW_HOST(OP_NAME, 'B');
                //.args(OP_PROPS);
        NEW_HOST(ssprintf("%d", pid), 't')
                .cat("OpDispatch")
                .id(OP_ID)
                .scope(std::to_string(pid));
        INC_COUNTER(op_execute_count, 1);
    });

    HANDLE_EVENT(OpExecuteFinishEvent, {
        auto& op_state = state.operators[event.op_id];
        op_state.host_end = time;
        NEW_HOST(OP_NAME, 'E')
                .args(OP_PROPS);
    });

    HANDLE_EVENT(KernelExecuteEvent, {
        auto& op_state = state.operators[event.op_id];
        op_state.device_begin = event.event;
        NEW_HOST(ssprintf("%d", pid), 's')
                .id(event.kernel_id)
                .cat("KernelLaunch")
                .scope(std::to_string(pid));
        NEW_DEVICE(OP_KERNEL_NAME, 'B')
                .cat("Kernel");
                //.args(OP_PROPS);
        NEW_DEVICE(ssprintf("%d", pid), 'f')
                .id(event.kernel_id)
                .bp('e')
                .cat("KernelLaunch")
                .scope(std::to_string(pid));
    });

    HANDLE_EVENT(KernelExecuteFinishEvent, {
        auto& op_state = state.operators[event.op_id];
        op_state.device_end = event.event;
        NEW_DEVICE(OP_KERNEL_NAME, 'E')
                .cat("Kernel")
                .args(OP_PROPS);
    });

    HANDLE_EVENT(TensorDeclareEvent, {
        auto& tensor_state = state.tensors[event.tensor_id] = {};
        tensor_state.id = event.tensor_id;
        tensor_state.name = event.name;
    });

    HANDLE_EVENT(TensorProduceEvent, {
        auto& tensor_state = state.tensors[event.tensor_id];
        tensor_state.device = event.device;
        tensor_state.layout = event.layout;
        tensor_state.produced = time;
        if (!tensor_state.living_time) {
            NEW_HOST(ssprintf("%d", pid), 's')
                .id(event.tensor_id)
                .cat("TensorLink")
                .scope(std::to_string(pid));
        } else {
            NEW_HOST(ssprintf("%d", pid), 't')
                .id(event.tensor_id)
                .cat("TensorLink")
                .scope(std::to_string(pid));
        }
        INC_COUNTER(alive_tensor_count, 1);
        INC_COUNTER(produce_tensor_count, 1);
        state.tensors_by_size.insert({tensor_state.id, tensor_state.size_in_bytes()});
        state.tensors_by_produced.insert({tensor_state.id, tensor_state.produced});
    });

    HANDLE_EVENT(TensorUsageEvent, {
        NEW_HOST(ssprintf("%d", pid), 't')
                .id(event.tensor_id)
                .cat("TensorLink")
                .scope(std::to_string(pid));
    });

    HANDLE_EVENT(TensorReleaseEvent, {
        auto& tensor_state = state.tensors[event.tensor_id];
        tensor_state.living_time += time - tensor_state.produced;
        tensor_state.produced = 0;
        INC_COUNTER(alive_tensor_count, -1);
        INC_COUNTER(erase_tensor_count, 1);
        state.tensors_by_size.erase({tensor_state.id, tensor_state.size_in_bytes()});
        state.tensors_by_produced.erase({tensor_state.id, tensor_state.produced});
        NEW_HOST(ssprintf("%d", pid), 't')
                .id(event.tensor_id)
                .cat("TensorLink")
                .scope(std::to_string(pid));
    });

    HANDLE_EVENT(TensorEraseEvent, {
        auto& tensor_state = state.tensors[event.tensor_id];
        if (tensor_state.living_time) {
            NEW_HOST(ssprintf("%d", pid), 'f')
                .id(event.tensor_id)
                .bp('e')
                .cat("TensorLink")
                .scope(std::to_string(pid));
        }
        if (event.use_count == 0) {
            INC_COUNTER(redundant_tensor_count, 1);
        }
    });

    HANDLE_EVENT(TensorGetPropEvent, {
        auto& tensor_state = state.tensors[event.tensor_id];
        NEW_HOST("TensorGetProp", 'X')
                .dur(0).args(TENSOR_PROPS);
    });

    HANDLE_EVENT(TensorWaitPropEvent, {
        NEW_HOST("TensorWaitProp", 'B');
        if (event.prop == TensorProp::HostValue) {
            INC_COUNTER(wait_value_count, 1);
        } else if (event.prop == TensorProp::Shape) {
            INC_COUNTER(wait_shape_count, 1);
        }
        INC_COUNTER(wait_prop_count, 1);
    });

    HANDLE_EVENT(TensorWaitPropFinishEvent, {
        auto& tensor_state = state.tensors[event.tensor_id];
        if (event.notified) {
            NEW_HOST(ssprintf("%d", pid), 'f')
                .id(event.tensor_id)
                .bp('e')
                .cat("TensorProp")
                .scope(std::to_string(pid));
        }
        NEW_HOST("TensorWaitProp", 'E')
                .args(TENSOR_PROPS);
    });

    HANDLE_EVENT(TensorNotifyPropEvent, {
        NEW_HOST(ssprintf("%d", pid), 's')
                .id(event.tensor_id)
                .cat("TensorProp")
                .scope(std::to_string(pid));
    });

    HANDLE_EVENT(ShapeInferEvent, {
        if (event.success) {
            INC_COUNTER(infer_shape_valid_count, 1);
        } else {
            INC_COUNTER(infer_shape_invalid_count, 1);
        }
    });

    HANDLE_EVENT(SampleDeviceEvent, {
        NEW_HOST("TopKTensor", 'B');
    });

    HANDLE_EVENT(SampleDeviceFinishEvent, {
        std::string device_name = event.device.locator().to_string();
        std::string prop_name = ssprintf("%s_alloc_memory", device_name.c_str());
        NEW_HOST(prop_name, 'C')
                .arg(prop_name, event.total_memory - event.free_memory);
        auto top_k_tensors = state.top_k_tensor_in_device(event.device, options.at("num_tensor_watch"));
        auto& top_k_event = NEW_HOST("TopKTensor", 'E');
        for (size_t i = 0; i < top_k_tensors.size(); ++i) {
            auto tensor_id = top_k_tensors[i];
            auto& tensor_state = state.tensors[tensor_id];
            top_k_event.arg(ssprintf("top%03d", (int)i), TENSOR_INFO); //%03d is always enough
        }
    });

    HANDLE_EVENT(WorkerExceptionEvent, {
        INC_COUNTER(exception_count, 1);
    });

    HANDLE_EVENT(TensorCommandEvent, {
        NEW_HOST(ssprintf("%s %zu", TENSOR_COMMAND_KIND, event.tensor_id), 'B');
    });

    HANDLE_EVENT(TensorCommandFinishEvent, {
        auto& tensor_state = state.tensors[event.tensor_id];
        NEW_HOST(ssprintf("%s %zu", TENSOR_COMMAND_KIND, event.tensor_id), 'E')
            .args(TENSOR_PROPS);
    });

    HANDLE_EVENT(ScopeEvent, {
        NEW_HOST(event.name, 'B');
        state.threads[tid].scope_stack.push_back(event.name);
    });

    HANDLE_EVENT(ScopeFinishEvent, {
        NEW_HOST(event.name, 'E');
        mgb_assert(state.threads[tid].scope_stack.back() == event.name);
        state.threads[tid].scope_stack.pop_back();
    });

    HANDLE_TENSOR_EVENT(OpInputEvent, OpInputFinishEvent, ssprintf("Input %zu", event.tensor_id));
    HANDLE_TENSOR_EVENT(OpOutputEvent, OpOutputFinishEvent, ssprintf("Output %zu", event.tensor_id));
    HANDLE_TENSOR_EVENT(OpDelEvent, OpDelFinishEvent, ssprintf("Del %zu", event.tensor_id));
    HANDLE_PLAIN_EVENT(StartProfileEvent, StartProfileFinishEvent, "StartProfile");
    HANDLE_PLAIN_EVENT(StopProfileEvent, StopProfileFinishEvent, "StopProfile");
    HANDLE_PLAIN_EVENT(CustomEvent, CustomFinishEvent, event.title);
    HANDLE_PLAIN_EVENT(AutoEvictEvent, AutoEvictFinishEvent, "AutoEvict");

    if (results.size() > 0) {
        uint64_t time = results[0].second.time;
        trace_events.new_event().name("Metadata").ph('I').pid(pid).ts(0).arg("localTime", time/1e3);
    }

    for (auto&& result: results) {
        collector(result.second.id, result.first, result.second.time, result.second.data);
    }

    for (auto&& [tid, thread]: state.threads) {
        if (!thread_dict.count(tid)) {
            continue;
        }
        trace_events.new_event().ts(0).name("thread_name").ph('M').pid(pid).tid(thread.index).arg("name", thread_dict[tid]);
    }

    for (auto&& [device, device_state]: state.devices) {
        trace_events.new_event().ts(0).name("thread_name").ph('M').pid(pid).tid(256+device_state.index).arg("name", device.to_string());
    }

    trace_events.to_json()->writeto_fpath(filename);

}

}
