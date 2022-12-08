#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>

#include <any>
#include <set>
#include <sstream>
#include <typeindex>

#include "nlohmann/json.hpp"

#include "megbrain/tensor.h"

#include "./events.h"

namespace mgb::imperative::profiler {

using StackManager = interpreter::intl::StackManager;

struct ProfileTensorState {
    uint64_t id = 0;
    std::optional<uint64_t> source;
    TensorLayout layout;
    CompNode device;
    std::string name;
    profiler::HostTime produced = profiler::HostTime::min();
    profiler::Duration living_time = profiler::Duration::zero();

    size_t size_in_bytes() const {
        if (!layout.dtype.valid()) {
            return 0;
        }
        return layout.dtype.size(layout.total_nr_elems());
    }

    std::string info(HostTime current_time) {
        std::string shape = layout.TensorShape::to_string();
        std::string dtype = layout.dtype.name();
        return ssprintf(
                "%s(%s:%s:%s)", name.c_str(), shape.c_str(), dtype.c_str(),
                device.to_string().c_str());
    }

    nlohmann::json detail(HostTime current_time) {
        nlohmann::json args;
        args["id"] = id;
        args["name"] = name;
        args["shape"] = layout.TensorShape::to_string();
        args["dtype"] = layout.dtype.name();
        args["nr_elements"] = layout.total_nr_elems();
        args["device"] = device.to_string();
        if (produced != produced.min()) {
            double ms_count = std::chrono::duration_cast<
                                      std::chrono::duration<double, std::micro>>(
                                      current_time - produced + living_time)
                                      .count();
            args["living_time"] = ssprintf("%lf ms", ms_count);
        }
        return args;
    }
};

struct ProfileOperatorState {
    uint64_t id = 0;
    std::string name;
    OpParams params;
    SmallVector<uint64_t> inputs;
    SmallVector<uint64_t> outputs;
    CompNode device;
    Trace trace;

    struct Execution {
        std::string reason;
        profiler::HostTime begin;
        profiler::HostTime end;
    };

    SmallVector<Execution> executions;

    nlohmann::json detail() {
        nlohmann::json args;
        for (auto&& [name, value] : params) {
            args[name] = value;
        }
        args["__id__"] = id;
        args["__name__"] = name;
        args["__device__"] = device.to_string();
        return args;
    }
};

template <typename TProp>
struct ProfileTensorPropPair {
    uint64_t id;
    TProp value;

    bool operator<(const ProfileTensorPropPair& lhs) const {
        return value == lhs.value ? id < lhs.id : value < lhs.value;
    }

    bool operator==(const ProfileTensorPropPair& lhs) const {
        return id == lhs.id && value == lhs.value;
    }

    bool operator>(const ProfileTensorPropPair& lhs) const {
        return value == lhs.value ? id > lhs.id : value > lhs.value;
    }
};

using ProfileTensorSizePair = ProfileTensorPropPair<size_t>;
using ProfileTensorProducedPair = ProfileTensorPropPair<uint64_t>;

struct ProfileState {
    std::unordered_map<uint64_t, ProfileTensorState> tensors;
    std::unordered_map<uint64_t, ProfileOperatorState> operators;
    std::unordered_map<std::string, uint64_t> tensor_name_counter;
    std::set<ProfileTensorSizePair> tensors_by_size;
    std::set<ProfileTensorSizePair> tensors_by_produced;

    std::vector<uint64_t> top_k_tensor_in_device(CompNode device, size_t k) {
        std::vector<uint64_t> results;
        for (auto iter = tensors_by_size.rbegin(); iter != tensors_by_size.rend();
             ++iter) {
            if (!k) {
                break;
            }
            if (tensors[iter->id].device == device) {
                results.push_back(iter->id);
                --k;
            }
        }
        return results;
    }
};

template <typename T, typename = void>
struct is_op_event : std::false_type {};

template <typename T>
struct is_op_event<T, decltype(std::declval<T>().op_id, void())> : std::true_type {};

template <typename T, typename = void>
struct is_tensor_event : std::false_type {};

template <typename T>
struct is_tensor_event<T, decltype(std::declval<T>().tensor_id, void())>
        : std::true_type {};
template <typename T, typename = void>
struct is_trace_event : std::false_type {};
template <typename T>
struct is_trace_event<T, decltype(std::declval<T>().trace, void())> : std::true_type {};

template <typename... TItems>
class AnyToVariantConverter {
public:
    using any_t = AnyPtr;
    using variant_t = std::variant<TItems...>;

private:
    std::unordered_map<std::type_index, std::function<variant_t(const any_t&)>> m_table;

    template <typename TItem>
    void register_converter() {
        m_table[typeid(TItem)] = [](const any_t& input) {
            return variant_t(input.cast<TItem>());
        };
    }

public:
    AnyToVariantConverter() { (register_converter<TItems>(), ...); }
    variant_t operator()(const any_t& input) {
        return m_table[input.type()](std::move(input));
    }
};

template <typename TSelf>
class EventVisitor {
private:
    std::unordered_map<size_t, ProfileOperatorState> m_operators;
    std::unordered_map<size_t, ProfileTensorState> m_tensors;
    std::unordered_map<size_t, std::vector<Profiler::Record>> m_duration_stack;
    HostTime m_start_time;
    CompNode::UnorderedMap<size_t> m_device_tid_table;
    std::unordered_map<std::thread::id, size_t> m_host_tid_table;
    std::unordered_map<cupti::stream_t, size_t> m_cupti_tid_table;
    CompNode::UnorderedMap<std::map<profiler::HostTime, profiler::RealDuration>>
            m_device_timeline;
    std::unordered_map<std::thread::id, std::vector<Trace>> m_trace_stack;
    std::unordered_map<std::string, int64_t> m_counter_table;
    std::optional<std::pair<profiler::HostTime, cupti::time_point>> m_cupti_timestamp =
            {};
    // Record the start and end time of all kernels
    std::vector<std::vector<profiler::HostTime>> m_kernel_start_finish_time;
    // The sum of all kernel execution times
    profiler::Duration m_gpu_usage_time = std::chrono::microseconds(0);
    // The sum of all kernel execution times, including the gap time between kernels
    // within a step
    profiler::Duration m_gpu_usage_time_with_gap = std::chrono::microseconds(0);
    // Record the end time of each step
    std::vector<profiler::HostTime> m_step_finish_time;
    // Record the start time of the first kernel and the end time of the last kernel in
    // each step
    std::vector<std::vector<profiler::HostTime>> m_step_first_last_kernel;

protected:
    Profiler::Record* current;
    ProfileOperatorState* current_op;
    ProfileTensorState* current_tensor;

protected:
    size_t next_tid() {
        return m_host_tid_table.size() + m_device_tid_table.size() +
               m_cupti_tid_table.size();
    }

    profiler::Duration since_start(profiler::HostTime time) {
        return time - m_start_time;
    }

    profiler::HostTime to_device_time(profiler::HostTime time, CompNode device) {
        auto& device_timeline = m_device_timeline[device];
        auto upper = device_timeline.lower_bound(time);
        if (upper == device_timeline.end()) {
            if (upper == device_timeline.begin()) {
                return time;
            } else {
                --upper;
                return time +
                       std::chrono::duration_cast<profiler::Duration>(upper->second);
            }
        } else if (upper->first == time) {
            return time + std::chrono::duration_cast<profiler::Duration>(upper->second);
        } else if (upper == device_timeline.begin()) {
            return time + std::chrono::duration_cast<profiler::Duration>(upper->second);
        }
        auto lower = upper;
        --lower;
        double ratio =
                ((double)(time - lower->first).count() /
                 (double)(upper->first - lower->first).count());
        mgb_assert(ratio > 0 && ratio < 1, "invalid ratio");
        mgb_assert(
                lower->first + lower->second <= upper->first + upper->second,
                "device time corr");
        auto shift = lower->second + ratio * (upper->second - lower->second);
        auto result = time + std::chrono::duration_cast<profiler::Duration>(shift);
        return result;
    }

    size_t to_tid(std::thread::id host_tid) { return m_host_tid_table.at(host_tid); }

    size_t to_tid(CompNode device) { return m_device_tid_table.at(device); }

    size_t to_tid(cupti::stream_t cupti_stream) {
        return m_cupti_tid_table.at(cupti_stream);
    }

    SmallVector<std::thread::id> host_threads() {
        SmallVector<std::thread::id> host_threads;
        for (auto&& [host, _] : m_host_tid_table) {
            host_threads.push_back(host);
        }
        return host_threads;
    }

    SmallVector<CompNode> devices() {
        SmallVector<CompNode> devices;
        for (auto&& [device, _] : m_device_tid_table) {
            devices.push_back(device);
        }
        return devices;
    }

    void inc_counter(const char* key, int64_t delta) {
        if (!m_counter_table.count(key)) {
            m_counter_table[key] = 0;
        }
        auto& value = m_counter_table[key];
        static_cast<TSelf&>(*this).notify_counter(key, value, value + delta);
        value += delta;
    }

    profiler::HostTime time_from_cupti(cupti::time_point timestamp) {
        mgb_assert(m_cupti_timestamp.has_value());
        return m_cupti_timestamp->first +
               std::chrono::duration_cast<profiler::HostTime::duration>(
                       timestamp - m_cupti_timestamp->second);
    }

public:
    void process_events(Profiler::bundle_t& bundle) {
        m_start_time = bundle.start_at;

        auto& self = static_cast<TSelf&>(*this);
        AnyToVariantConverter<
                OpDispatchEvent, OpExecuteEvent, OpExecuteFinishEvent,
                KernelLaunchEvent, KernelLaunchFinishEvent, OpInputEvent,
                OpInputFinishEvent, OpOutputEvent, OpOutputFinishEvent,
                TensorDeclareEvent, TensorProduceEvent, TensorUsageEvent,
                TensorReleaseEvent, TensorEraseEvent, TensorGetPropEvent,
                TensorNotifyPropEvent, TensorWaitPropEvent, TensorWaitPropFinishEvent,
                SampleDeviceEvent, SampleDeviceFinishEvent, WorkerExceptionEvent,
                ShapeInferEvent, SyncEvent, SyncFinishEvent, StartProfileEvent,
                StartProfileFinishEvent, StopProfileEvent, StopProfileFinishEvent,
                StopStepEvent, TensorCommandEvent, TensorCommandFinishEvent,
                AutoEvictEvent, AutoEvictFinishEvent, CustomEvent, CustomFinishEvent,
                RecordDeviceEvent, ScopeEvent, ScopeFinishEvent, HostToDeviceEvent,
                HostToDeviceFinishEvent, CUPTITimestampEvent, CUPTIKernelLaunchEvent,
                CUPTIKernelLaunchFinishEvent, CUPTIKernelExecuteEvent,
                CUPTIMemcpyLaunchEvent, CUPTIMemcpyLaunchFinishEvent, CUPTIMemcpyEvent,
                CUPTIRuntimeEvent, CUPTIRuntimeFinishEvent, CUPTIDriverEvent,
                CUPTIDriverFinishEvent, CUPTIMemsetEvent>
                converter;

        auto for_each_entry = [&](auto&& handler) {
            for (auto& entry : bundle.entries) {
                current = &entry;
                std::visit(handler, converter(entry.data));
            }
            current = nullptr;
        };

        // build device timeline
        struct DeviceStartPair {
            profiler::HostTime host;
            std::shared_ptr<CompNode::Event> device;
        };
        CompNode::UnorderedMap<DeviceStartPair> device_start_table;
        std::unordered_map<cupti::stream_t, CompNode> cupti_stream_table;

        // record device time
        for_each_entry([&](auto&& event) {
            using T = std::decay_t<decltype(event)>;
            if constexpr (std::is_same_v<T, RecordDeviceEvent>) {
                using namespace std::chrono_literals;
                DeviceStartPair& device_start =
                        device_start_table[event.event->comp_node()];
                if (!device_start.device) {
                    device_start = {current->time, event.event};
                }
                event.event->host_wait();
                auto device_time =
                        (device_start.host - current->time) +
                        std::chrono::duration_cast<profiler::RealDuration>(
                                device_start.device->elapsed_time_until(*event.event) *
                                1s);
                m_device_timeline[event.event->comp_node()][current->time] =
                        device_time;
            }
        });

        // record step end time
        for_each_entry([&](auto&& event) {
            using T = std::decay_t<decltype(event)>;
            if constexpr (std::is_same_v<T, StopStepEvent>) {
                auto step_time = current->time;
                m_step_finish_time.push_back(to_device_time(step_time, event.device));
            }
        });

        // register host threads
        for_each_entry([&](auto&& event) {
            if (!m_host_tid_table.count(current->tid)) {
                m_host_tid_table[current->tid] = next_tid();
            }
        });

        for_each_entry([&](auto&& event) {
            using T = std::decay_t<decltype(event)>;
            if constexpr (std::is_same_v<T, OpDispatchEvent>) {
                auto& op = m_operators[event.op_id];
                mgb_assert(op.id == 0, "duplicate operator id");
                op.id = event.op_id;
                op.name = event.op_name;
                op.params = event.op_params();
                op.inputs = event.inputs;
                op.outputs = event.outputs;
                op.trace = event.trace;
                for (auto&& output : event.outputs) {
                    m_tensors[output].source = op.id;
                }
            } else if constexpr (std::is_same_v<T, TensorDeclareEvent>) {
                auto& tensor = m_tensors[event.tensor_id];
                mgb_assert(tensor.id == 0, "duplicated tensor id");
                tensor.id = event.tensor_id;
                tensor.name = event.name;
            } else if constexpr (std::is_same_v<T, TensorProduceEvent>) {
                auto& tensor = m_tensors[event.tensor_id];
                if (!m_device_tid_table.count(event.device)) {
                    m_device_tid_table[event.device] = next_tid();
                }
                tensor.device = event.device;
                tensor.layout = event.layout;
            }
        });

        for_each_entry([&](auto&& event) {
            using T = std::decay_t<decltype(event)>;
            if constexpr (std::is_same_v<T, CUPTIIdentifyStreamEvent>) {
                if (!m_cupti_tid_table.count(event.stream)) {
                    m_cupti_tid_table[event.stream] =
                            m_device_tid_table.at(event.device);
                }
            }
        });

        // record cupti streams
        for_each_entry([&](auto&& event) {
            using T = std::decay_t<decltype(event)>;
            if constexpr (
                    std::is_same_v<T, CUPTIKernelExecuteEvent> ||
                    std::is_same_v<T, CUPTIMemcpyEvent> ||
                    std::is_same_v<T, CUPTIMemsetEvent>) {
                if (!m_cupti_tid_table.count(event.stream)) {
                    m_cupti_tid_table[event.stream] = next_tid();
                }
            } else if constexpr (std::is_same_v<T, CUPTITimestampEvent>) {
                mgb_assert(!m_cupti_timestamp.has_value());
                m_cupti_timestamp.emplace(current->time, event.timestamp);
            }
        });

        // replay execution
        using namespace std::placeholders;
        for_each_entry([&](auto&& event) {
            using T = std::decay_t<decltype(event)>;
            // update current_op/tensor
            if constexpr (is_op_event<T>::value) {
                current_op = &m_operators[event.op_id];
                if (current_op->id == 0) {
                    current_op->id = event.op_id;
                    current_op->name = "UnknownOperator";
                }
            } else if constexpr (is_tensor_event<T>::value) {
                current_tensor = &m_tensors[event.tensor_id];
                if (current_tensor->id == 0) {
                    current_tensor->id = event.tensor_id;
                    current_tensor->name = "UnknownTensor";
                }
            }
            if constexpr (std::is_same_v<T, OpExecuteEvent>) {
                current_op->executions.emplace_back();
                current_op->executions.back().reason = event.reason;
                current_op->executions.back().begin = current->time;
            } else if constexpr (std::is_same_v<T, OpExecuteFinishEvent>) {
                current_op->executions.back().end = current->time;
            }
            // update counters
            if constexpr (std::is_same_v<T, OpDispatchEvent>) {
                inc_counter("nr_op_pending", 1);
            } else if constexpr (std::is_same_v<T, OpExecuteEvent>) {
                inc_counter("nr_op_pending", -1);
            } else if constexpr (std::is_same_v<T, TensorProduceEvent>) {
                inc_counter("nr_alive_tensor", 1);
            } else if constexpr (std::is_same_v<T, TensorReleaseEvent>) {
                inc_counter("nr_alive_tensor", -1);
            } else if constexpr (std::is_same_v<T, TensorEraseEvent>) {
                if (event.use_count == 0) {
                    inc_counter("nr_redunant_tensor", 1);
                }
            } else if constexpr (std::is_same_v<T, ShapeInferEvent>) {
                if (!event.success) {
                    inc_counter("nr_shape_infer_failure", 1);
                }
            } else if constexpr (std::is_same_v<T, WorkerExceptionEvent>) {
                inc_counter("nr_exception", 1);
            } else if constexpr (std::is_same_v<T, KernelLaunchFinishEvent>) {
                auto& execution = current_op->executions.back();
                auto overhead = to_device_time(current->time, event.device) -
                                to_device_time(execution.begin, event.device);

                std::vector<profiler::HostTime> current_kernel_start_finish;
                current_kernel_start_finish.emplace_back(
                        to_device_time(execution.begin, event.device));
                current_kernel_start_finish.emplace_back(
                        to_device_time(current->time, event.device));
                m_kernel_start_finish_time.emplace_back(current_kernel_start_finish);

                if (execution.reason == "dtr") {
                    inc_counter(
                            "dtr_overhead_us",
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                    overhead)
                                    .count());
                }
            }
            // visit_event_impl
            self.visit_event(event);
            // reset current_op/tensor
            if constexpr (is_op_event<T>::value) {
                current_op = nullptr;
            } else if constexpr (is_tensor_event<T>::value) {
                current_tensor = nullptr;
            }
        });
    }

    profiler::Duration last_kernel_finish() {
        if (m_kernel_start_finish_time.size() == 0) {
            return profiler::Duration::zero();
        }
        return m_kernel_start_finish_time.back()[1] - m_start_time;
    }

    // get GPU busy time (union of calculation time and communication time)
    profiler::Duration gpu_usage_time() {
        if (m_kernel_start_finish_time.size() == 0) {
            return profiler::Duration::zero();
        }

        std::sort(
                m_kernel_start_finish_time.begin(), m_kernel_start_finish_time.end(),
                [&](std::vector<profiler::HostTime> kernel1,
                    std::vector<profiler::HostTime> kernel2) {
                    if (kernel1[0] != kernel2[0]) {
                        return kernel1[0] < kernel2[0];
                    }
                    return kernel1[1] < kernel2[1];
                });

        HostTime current_start = profiler::HostTime::min();
        HostTime current_end = profiler::HostTime::min();
        for (size_t i = 0; i < m_kernel_start_finish_time.size(); ++i) {
            if (current_start == profiler::HostTime::min()) {
                current_start = m_kernel_start_finish_time[i][0];
                current_end = m_kernel_start_finish_time[i][1];
            } else if (current_end < m_kernel_start_finish_time[i][0]) {
                m_gpu_usage_time += profiler::Duration(current_end - current_start);
                current_start = m_kernel_start_finish_time[i][0];
                current_end = m_kernel_start_finish_time[i][1];
            } else if (current_end > m_kernel_start_finish_time[i][0]) {
                current_end = max(current_end, m_kernel_start_finish_time[i][1]);
            }
        }
        m_gpu_usage_time += profiler::Duration(current_end - current_start);

        return m_gpu_usage_time;
    }

    // compared to gpu_usage_time, this method adds gap time between kernels in the same
    // step
    profiler::Duration gpu_usage_time_with_gap() {
        if (m_step_finish_time.empty()) {
            return profiler::Duration::zero();
        }

        std::sort(
                m_step_finish_time.begin(), m_step_finish_time.end(),
                [&](HostTime time1, HostTime time2) { return time1 < time2; });

        std::sort(
                m_kernel_start_finish_time.begin(), m_kernel_start_finish_time.end(),
                [&](std::vector<profiler::HostTime> kernel1,
                    std::vector<profiler::HostTime> kernel2) {
                    if (kernel1[0] != kernel2[0]) {
                        return kernel1[0] < kernel2[0];
                    }
                    return kernel1[1] < kernel2[1];
                });

        int cur_step = 0;
        auto kernel_num = m_kernel_start_finish_time.size();
        for (size_t i = 0; i < kernel_num; ++i) {
            // Record the start time of the first kernel and the end time of the last
            // kernel of the current step
            std::vector<profiler::HostTime> step_begin_end_time;
            step_begin_end_time.emplace_back(m_kernel_start_finish_time[i][0]);
            size_t j = i;
            while (j < kernel_num && m_kernel_start_finish_time[j][0] <
                                             m_step_finish_time[cur_step + 1]) {
                ++j;
            }
            step_begin_end_time.emplace_back(m_kernel_start_finish_time[j - 1][1]);
            mgb_assert(
                    step_begin_end_time.size() == 2,
                    "step_begin_end_time.size() should be exactly 2!");
            m_step_first_last_kernel.emplace_back(step_begin_end_time);
            i = j - 1;
            ++cur_step;
        }

        for (size_t i = 0; i < m_step_first_last_kernel.size(); ++i) {
            m_gpu_usage_time_with_gap += profiler::Duration(
                    m_step_first_last_kernel[i][1] - m_step_first_last_kernel[i][0]);
        }

        return m_gpu_usage_time_with_gap;
    }

    // from the start time of the first kernel of the first step to the end time of the
    // last kernel of the last step
    std::chrono::microseconds get_total_train_time() {
        if (m_kernel_start_finish_time.size() == 0) {
            return std::chrono::microseconds(1);
        }
        return std::chrono::duration_cast<std::chrono::microseconds>(
                m_kernel_start_finish_time.back()[1] -
                m_kernel_start_finish_time.front()[0]);
    }
};

}  // namespace mgb::imperative::profiler
