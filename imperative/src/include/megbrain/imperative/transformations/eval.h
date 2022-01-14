/**
 * \file imperative/src/include/megbrain/imperative/eval.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/helper.h"

namespace mgb::imperative {

struct InterpreterInfo {
public:
    using Handle = interpreter::Interpreter::Handle;
    using Channel = interpreter::Interpreter::Channel;

private:
    std::shared_ptr<Handle> m_handle = nullptr;
    std::string m_name;

public:
    InterpreterInfo() = default;
    InterpreterInfo(std::shared_ptr<Handle> handle, std::string name = {})
            : m_handle(handle), m_name(name) {}

    std::shared_ptr<Handle> handle() const { return m_handle; }

    std::string name() const { return m_name; }
};

class InterpreterValue final
        : public MixinValueImpl<InterpreterValue, InterpreterInfo> {
public:
    using MixinValueImpl::MixinValueImpl;

    std::string to_string() const override {
        return ssprintf(
                "Handle{ptr=%p, name=%s}", handle().get(),
                imperative::quoted(name()).c_str());
    }
};

/**
 * \brief interpret operations with interpreter
 *
 * This is the most basic and simplest transformation. It read operation requests and
 * forwards them to interpreter. Not all tensor requests would be handled by it,
 * some were resolved by CompiledTransformation or LazyEvalTransformation.
 */
class InterpreterTransformation final : public Transformation {
public:
    using Interpreter = interpreter::Interpreter;
    using Handle = Interpreter::Handle;
    using Channel = Interpreter::Channel;

private:
    std::unique_ptr<Channel> m_channel;

public:
    explicit InterpreterTransformation(std::unique_ptr<Channel> channel)
            : m_channel{std::move(channel)} {}

    Channel* channel() { return m_channel.get(); }

    std::vector<ValueRef> apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is<InterpreterValue>());
        return value;
    }

    std::string name() const override { return "InterpreterTransformation"; }

    std::shared_ptr<Handle> share_handle(Handle handle) {
        return std::shared_ptr<Handle>(
                new Handle(handle), [channel = m_channel.get()](Handle* ptr) {
                    if (ptr) {
                        channel->del(*ptr);
                        delete ptr;
                    }
                });
    }
};

}  // namespace mgb::imperative
