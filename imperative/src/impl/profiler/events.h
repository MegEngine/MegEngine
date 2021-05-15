/**
 * \file imperative/src/impl/interpreter/events.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/small_vector.h"

#include "../op_trait.h"

namespace mgb::imperative::profiler {

enum class TensorProp {
    InvalidProp, Device, Shape, DType, DevValue, HostValue,
};

using OpParams = std::unordered_map<std::string, std::string>;

}

namespace mgb::imperative {

template <>
struct ToStringTrait<profiler::TensorProp>{
    using TensorProp = profiler::TensorProp;
    std::string operator()(TensorProp prop) const {
        switch(prop) {
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

}

namespace mgb::imperative::profiler {

#define DEF_EVENT(X, ...) struct X##Event __VA_ARGS__;
#define DEF_DUR_EVENT(X, ...) struct X##Event __VA_ARGS__; struct X##FinishEvent __VA_ARGS__;

DEF_EVENT(OpDispatch, {
    uint64_t op_id;
    std::string op_name;
    std::function<OpParams()> op_params;
    SmallVector<uint64_t> inputs;
    SmallVector<uint64_t> outputs;
});

DEF_DUR_EVENT(OpInput, {
    uint64_t tensor_id;
    TensorShape shape;
});

DEF_DUR_EVENT(OpDel, {
    uint64_t tensor_id;
    TensorShape shape;
});

DEF_DUR_EVENT(OpOutput, {
    uint64_t tensor_id;
    TensorShape shape;
});

DEF_DUR_EVENT(OpExecute, {
    uint64_t op_id;
});

DEF_DUR_EVENT(OpPostExecute, {
    uint64_t op_id;
});

DEF_DUR_EVENT(KernelExecute, {
    uint64_t op_id;
    uint64_t kernel_id;
    std::shared_ptr<CompNode::Event> event;
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

DEF_EVENT(TensorUsage, {
    uint64_t tensor_id;
});

DEF_EVENT(TensorRelease, {
    uint64_t tensor_id;
});

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

DEF_EVENT(TensorWaitProp, {
    uint64_t tensor_id;
    uint64_t wait_id;
    TensorProp prop;
});

DEF_EVENT(TensorWaitPropFinish, {
    uint64_t tensor_id;
    uint64_t wait_id;
    TensorProp prop;
    bool notified;
});

DEF_DUR_EVENT(SampleDevice, {
    CompNode device;
    size_t total_memory;
    size_t free_memory;
});

DEF_EVENT(WorkerException, {});

DEF_EVENT(ShapeInfer, {
    bool success;
});

DEF_DUR_EVENT(Scope, {
    std::string name;
});

DEF_DUR_EVENT(DeviceScope, {
    std::string name;
    std::shared_ptr<CompNode::Event> event;
});

DEF_DUR_EVENT(Sync, {});

DEF_DUR_EVENT(StartProfile, {
    size_t capture_count;
});

DEF_DUR_EVENT(StopProfile, {
    size_t escape_count;
});

DEF_DUR_EVENT(TensorCommand, {
    enum Kind {
        Put, Del, SwapIn, SwapOut, Drop, ReGen, RecFree, GetValue
    };
    uint64_t tensor_id;
    Kind kind;
});

DEF_DUR_EVENT(AutoEvict, {});

DEF_DUR_EVENT(Custom, {
    std::string title;
    std::string content;
});

#undef DEF_EVENT
#undef DEF_DUR_EVENT

}
