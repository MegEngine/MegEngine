#pragma once

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/helper.h"

namespace mgb::imperative {

class InterpreterValue final : public ObjectValue<InterpreterValue> {
public:
    using Handle = interpreter::Interpreter::Handle;
    using Channel = interpreter::Interpreter::Channel;

    class RAIIHandle : public NonCopyableObj {
    private:
        Handle m_handle = nullptr;
        Channel* m_channel = nullptr;

    public:
        RAIIHandle(Handle handle, Channel* channel)
                : m_handle(handle), m_channel(channel) {}
        ~RAIIHandle() { m_channel->del(m_handle); }

        Handle handle() const { return m_handle; }

        Channel* channel() const { return m_channel; }
    };

private:
    LocalPtr<RAIIHandle> m_handle;
    std::string m_name;
    mutable DTypeValue::ref_t m_dtype;
    mutable CompNodeValue::ref_t m_comp_node;
    mutable ShapeValue::ref_t m_shape;

public:
    InterpreterValue(LocalPtr<RAIIHandle> handle, std::string name = {})
            : m_handle(handle), m_name(name) {}

    const LocalPtr<RAIIHandle>& handle() const { return m_handle; }

    DTypeValue::ref_t dtype() const;
    CompNodeValue::ref_t comp_node() const;
    ShapeValue::ref_t shape() const;

    std::string name() const { return m_name; }

    std::string to_string() const override {
        return ssprintf(
                "Handle{ptr=%p, name=%s}", handle().get(),
                imperative::quoted(name()).c_str());
    }

    void clear() override { m_handle = {}; }
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
    using SharedHandle = LocalPtr<InterpreterValue::RAIIHandle>;
    using Channel = Interpreter::Channel;

private:
    std::shared_ptr<Channel> m_channel;
    ObjectType<InterpreterValue> m_value_type{"InterpreterValue"};

public:
    explicit InterpreterTransformation(std::shared_ptr<Channel> channel)
            : m_channel{std::move(channel)} {}

    Channel* channel() { return m_channel.get(); }

    ValueRefList apply_op(const ApplyOp& apply_op, Span<ValueRef> inputs);

    ValueRefList apply_get_attr(const GetAttr& get_attr, Span<ValueRef> inputs);

    ValueRefList apply_create_tensor(
            const CreateTensor& create_tensor, Span<ValueRef> inputs);

    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is(m_value_type));
        return value;
    }

    std::string name() const override { return "InterpreterTransformation"; }

    SharedHandle share_handle(Handle handle) {
        return SharedHandle::make(handle, m_channel.get());
    }
};

}  // namespace mgb::imperative
