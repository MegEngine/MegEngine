/**
 * \file imperative/src/impl/interpreter/profiler.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/profiler.h"

#include "./commands.h"
#include "./events.h"
#include "./option_manager.h"

namespace mgb::imperative::interpreter::intl {

class InterpreterProfiler: public Profiler<
        CommandEnqueueEvent, CommandExecuteEvent, CommandFinishEvent,
        OpExecuteEvent, OpExecuteFinishEvent,
        KernelExecuteEvent, KernelExecuteFinishEvent,
        TensorDeclareEvent, TensorProduceEvent, TensorEraseEvent,
        TensorGetPropEvent, TensorWaitPropEvent, TensorNotifyPropEvent, TensorWaitPropFinishEvent,
        SyncEvent, SyncFinishEvent,
        ScopeEvent, ScopeFinishEvent,
        DeviceScopeEvent, DeviceScopeFinishEvent> {

public:
    enum Topic {
        Command         = 0b000001,
        Operator        = 0b000010,
        TensorLifetime  = 0b000100,
        TensorProp      = 0b001000,
        Sync            = 0b010000,
        Scope           = 0b100000,
    };

    struct Option {
        Topic topic;
        bool align_time;
        bool show_operator_name;

        static Option from_dict(std::unordered_map<std::string, int> dict) {
            Option option;
            option.topic = Topic(dict.at("topic"));
            option.align_time = bool(dict.at("align_time"));
            option.show_operator_name = bool(dict.at("show_operator_name"));
            return option;
        }
    };

    Option get_option() const {
        return m_option;
    }

    void set_option(const Option& option) {
        m_option = option;
    }

    static Mask topic_to_mask(Topic topic) {
        Mask result;
        if (topic & Command) {
            result |= mask_of<CommandEnqueueEvent, CommandExecuteEvent, CommandFinishEvent>();
        }
        if (topic & Operator) {
            result |= mask_of<OpExecuteEvent, OpExecuteFinishEvent>();
            result |= mask_of<KernelExecuteEvent, KernelExecuteFinishEvent>();
        }
        if (topic & TensorLifetime) {
            result |= mask_of<TensorDeclareEvent, TensorProduceEvent, TensorEraseEvent>();
        }
        if (topic & TensorProp) {
            result |= mask_of<TensorGetPropEvent, TensorWaitPropEvent, TensorNotifyPropEvent, TensorWaitPropFinishEvent>();
        }
        if (topic & Sync) {
            result |= mask_of<SyncEvent, SyncFinishEvent>();
        }
        if (topic & Scope) {
            result |= mask_of<ScopeEvent, ScopeFinishEvent>();
            result |= mask_of<DeviceScopeEvent, DeviceScopeFinishEvent>();
        }
        return result;
    }

private:
    Option m_option;
};

}
