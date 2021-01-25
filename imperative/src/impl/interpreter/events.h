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

#include "./commands.h"
#include "./tensor_info.h"

namespace mgb::imperative::interpreter::intl {

struct CommandEvent {
    IdentifiedCommand icmd;
};

struct CommandEnqueueEvent: CommandEvent {};

struct CommandExecuteEvent: CommandEvent {};

struct CommandFinishEvent: CommandEvent {};

struct OpEvent {
    uint64_t id;
    std::shared_ptr<OpDef> op;
    SmallVector<uint64_t> inputs;
    SmallVector<uint64_t> outputs;
};

struct HostOpExecuteEvent: OpEvent {};

struct DeviceOpExecuteEvent: OpEvent {};

struct HostOpFinishEvent: OpEvent {};

struct DeviceOpFinishEvent: OpEvent {};

struct TensorDeclareEvent {
    uint64_t tensor_id;
};

struct TensorProduceEvent {
    uint64_t tensor_id;
    TensorLayout layout;
    CompNode device;
};

struct TensorEraseEvent {
    uint64_t tensor_id;
};

struct TensorPropEvent {
    uint64_t tensor_id;
    TensorInfo::Prop prop;
    std::string prop_desc;
};

struct TensorGetPropEvent: TensorPropEvent{};

struct TensorWaitPropEvent: TensorPropEvent{};

struct TensorNotifyPropEvent: TensorPropEvent{};

struct TensorWaitPropFinishEvent: TensorPropEvent{};

struct SyncStartEvent {};

struct SyncFinishEvent {};

struct ScopeEvent {
    std::string name;
};

struct ChannelBeginScope: ScopeEvent {};

struct ChannelEndScope: ScopeEvent {};

struct WorkerBeginScope: ScopeEvent {};

struct WorkerEndScope: ScopeEvent {};

struct DeviceBeginScope: ScopeEvent {};

struct DeviceEndScope: ScopeEvent {};

}
