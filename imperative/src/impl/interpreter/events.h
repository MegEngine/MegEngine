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

#define DEF_EVENT(X, ...) struct X##Event __VA_ARGS__;
#define DEF_DUR_EVENT(X, ...) struct X##Event __VA_ARGS__; struct X##FinishEvent __VA_ARGS__;

DEF_EVENT(Command, {
    IdentifiedCommand icmd;
});

DEF_EVENT(CommandEnqueue, :CommandEvent {});
DEF_EVENT(CommandExecute, :CommandEvent {});
DEF_EVENT(CommandFinish, :CommandEvent {});
DEF_DUR_EVENT(OpExecute, {
    uint64_t id;
    std::shared_ptr<OpDef> op;
    SmallVector<uint64_t> inputs;
    SmallVector<uint64_t> outputs;
});
DEF_DUR_EVENT(KernelExecute, {
    uint64_t id;
    std::shared_ptr<OpDef> op;
    SmallVector<uint64_t> inputs;
    SmallVector<uint64_t> outputs;
});
DEF_EVENT(TensorDeclare, {
    uint64_t tensor_id;
});
DEF_EVENT(TensorProduce, {
    uint64_t tensor_id;
    TensorLayout layout;
    CompNode device;
});
DEF_EVENT(TensorErase, {
    uint64_t tensor_id;
});
DEF_EVENT(TensorGetProp, {
    uint64_t tensor_id;
    TensorInfo::Prop prop;
    std::string prop_desc;
});
DEF_DUR_EVENT(TensorWaitProp, {
    uint64_t tensor_id;
    TensorInfo::Prop prop;
    std::string prop_desc;
});
DEF_EVENT(TensorNotifyProp, {
    uint64_t tensor_id;
    TensorInfo::Prop prop;
    std::string prop_desc;
});
DEF_DUR_EVENT(Sync, {});
DEF_DUR_EVENT(Scope, {
    std::string name;
});
DEF_DUR_EVENT(DeviceScope, {
    std::string name;
});

}
