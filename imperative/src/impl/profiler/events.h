#pragma once

#include "megbrain/imperative/profiler.h"
#include "megbrain/utils/small_vector.h"

#include "../interpreter/stack_manager.h"
#include "../op_trait.h"
#include "megbrain/imperative/cpp_cupti.h"

namespace mgb::imperative::profiler {

enum class TensorProp {
    InvalidProp,
    Device,
    Shape,
    DType,
    DevValue,
    HostValue,
};

using OpParams = std::unordered_map<std::string, std::string>;

}  // namespace mgb::imperative::profiler

namespace mgb::imperative {

template <>
struct ToStringTrait<profiler::TensorProp> {
    using TensorProp = profiler::TensorProp;
    std::string operator()(TensorProp prop) const {
        switch (prop) {
            case TensorProp::DType:
                return "dtype";
            case TensorProp::DevValue:
                return "dev_value";
            case TensorProp::Device:
                return "device";
            case TensorProp::HostValue:
                return "host_value";
            case TensorProp::Shape:
                return "shape";
            default:
                return "unknown";
        }
    }
};

}  // namespace mgb::imperative

namespace mgb::imperative::profiler {

using Trace = interpreter::intl::StackManager::Trace;

struct ProfileOperatorState;
struct ProfileTensorState;

#define DEF_EVENT(X, ...) struct X##Event __VA_ARGS__;
#define DEF_DUR_EVENT(X, ...)    \
    struct X##Event __VA_ARGS__; \
    struct X##FinishEvent __VA_ARGS__;

DEF_EVENT(OpDispatch, {
    uint64_t op_id;
    std::string op_name;
    std::function<OpParams()> op_params;
    SmallVector<uint64_t> inputs;
    SmallVector<uint64_t> outputs;
    Trace trace;
});

DEF_DUR_EVENT(OpInput, {
    uint64_t tensor_id;
    TensorShape shape;
});

DEF_DUR_EVENT(OpOutput, {
    uint64_t tensor_id;
    TensorShape shape;
});

DEF_DUR_EVENT(OpExecute, {
    uint64_t op_id;
    SmallVector<CompNode> device_list;
    std::string reason;
});

DEF_DUR_EVENT(KernelLaunch, {
    uint64_t op_id;
    uint64_t kernel_id;
    CompNode device;
});

DEF_EVENT(TensorDeclare, {
    uint64_t tensor_id;
    std::string name;
});

DEF_EVENT(TensorProduce, {
    uint64_t tensor_id;
    TensorLayout layout;
    CompNode device;
    void* ptr;
});

DEF_EVENT(TensorUsage, { uint64_t tensor_id; });

DEF_EVENT(TensorRelease, { uint64_t tensor_id; });

DEF_EVENT(TensorErase, {
    uint64_t tensor_id;
    size_t use_count;
});

DEF_EVENT(TensorGetProp, {
    uint64_t tensor_id;
    TensorProp prop;
});

DEF_EVENT(TensorNotifyProp, {
    uint64_t tensor_id;
    uint64_t wait_id;
    TensorProp prop;
});

DEF_DUR_EVENT(TensorWaitProp, {
    uint64_t tensor_id;
    uint64_t wait_id;
    TensorProp prop;
    std::function<OpParams()> param;
});

DEF_DUR_EVENT(SampleDevice, {
    CompNode device;
    size_t total_memory;
    size_t free_memory;
});

DEF_EVENT(WorkerException, {});

DEF_EVENT(ShapeInfer, { bool success; });

DEF_DUR_EVENT(Scope, { std::string name; });

DEF_DUR_EVENT(Sync, { Trace trace; });

DEF_DUR_EVENT(StartProfile, { size_t capture_count; });

DEF_DUR_EVENT(StopProfile, { size_t escape_count; });

enum class TensorCommandKind { Put, Del, Drop, ReGen, RecFree, GetValue };

DEF_DUR_EVENT(TensorCommand, {
    using Kind = TensorCommandKind;
    uint64_t tensor_id;
    Kind kind;
});

DEF_DUR_EVENT(AutoEvict, {});

DEF_DUR_EVENT(Custom, {
    std::string title;
    std::string content;
    CompNode device;
});

DEF_EVENT(RecordDevice, { std::shared_ptr<CompNode::Event> event; });

DEF_DUR_EVENT(HostToDevice, {
    TensorLayout layout;
    CompNode device;
    void* host_ptr;
    void* device_ptr;
});

// cupti events
DEF_EVENT(CUPTITimestamp, { cupti::clock::time_point timestamp; });

DEF_DUR_EVENT(CUPTIKernelLaunch, {
    uint32_t correlation_id;
    const char* name;
});

DEF_EVENT(CUPTIKernelExecute, {
    uint32_t correlation_id;
    const char* name;
    cupti::stream_t stream;
    cupti::time_point start;
    cupti::time_point end;
});

DEF_DUR_EVENT(CUPTIMemcpyLaunch, { uint32_t correlation_id; });

DEF_EVENT(CUPTIMemcpy, {
    uint32_t correlation_id;
    uint8_t src_kind;
    uint8_t dst_kind;
    uint64_t bytes;
    cupti::stream_t stream;
    cupti::time_point start;
    cupti::time_point end;
});

DEF_EVENT(CUPTIMemset, {
    uint32_t correlation_id;
    uint32_t value;
    uint64_t bytes;
    cupti::stream_t stream;
    cupti::time_point start;
    cupti::time_point end;
});

DEF_EVENT(CUPTIUnknownDevice, {});

DEF_DUR_EVENT(CUPTIRuntime, {
    uint32_t correlation_id;
    const char* name;
});

DEF_DUR_EVENT(CUPTIDriver, {
    uint32_t correlation_id;
    const char* name;
});

DEF_EVENT(CUPTIIdentifyStream, {
    cupti::stream_t stream;
    CompNode device;
});

#undef DEF_EVENT
#undef DEF_DUR_EVENT

}  // namespace mgb::imperative::profiler
