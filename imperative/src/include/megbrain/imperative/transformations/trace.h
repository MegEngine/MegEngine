/**
 * \file imperative/src/include/megbrain/imperative/trace.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <chrono>
#include <future>
#include <variant>

#include "megbrain/gopt/inference.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/utils/box.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/serializer.h"

namespace mgb::imperative {

struct TraceResult {
    struct SeqItem {
        std::shared_ptr<OpDef> op;
        SmallVector<size_t> inputs;
        SmallVector<size_t> outputs;
    };

    struct VarInfo {
        enum Kind {
            External,  // End point of traced graph, its value is received from
                       // environment
            Constant,  // Also end point, but its value is constant in all executions,
                       // so we don't need to get from env every time, just capture it
            Internal,  // Not end point, produced by some op (or just forwarded) from
                       // op_seq
        };

        size_t id;
        DTypeValue::ref_t dtype;
        CompNodeValue::ref_t device;

        // if exists, for input: assert equal
        // for output: get_data/shape/value
        ValueRef bound_data;
        std::string mark;
        std::string name;

        Kind kind;
        bool value_required = false;
        bool data_required = false;
        bool shape_required = false;

        TensorShape shape;
    };

    using VarKind = VarInfo::Kind;

    std::vector<SeqItem> seq;
    std::vector<VarInfo> vars;

    /**
     * \brief dump to mgb computing graph
     *
     * \param graph mgb computing graph
     * \param inputs (input_id, input_name, input_shape)
     * \param outputs (output_id, outupt_name)
     * \param prefer_input_names
     * \return VarNodeArray output nodes
     */
    VarNodeArray dump(
            ComputingGraph& graph,
            std::vector<std::tuple<size_t, std::string, TensorShape>> inputs,
            std::vector<std::pair<size_t, std::string>> outputs,
            bool prefer_input_names);
};

/**
 * \brief mark an var as arg/kwarg/output
 *
 */
class TraceMarkVar : public OperatorImpl<TraceMarkVar, Operator::IdentityLike> {
private:
    std::string m_mark;

public:
    TraceMarkVar(std::string mark) : m_mark(mark) {}

    std::string mark() const { return m_mark; }

    std::string to_string() const override {
        return ssprintf("TraceMarkVar{mark=%s}", imperative::quoted(m_mark).c_str());
    }
};

class TracingInfo {
private:
    ValueRef m_value = {};
    size_t m_id = 0;

public:
    TracingInfo() = default;
    TracingInfo(ValueRef value, size_t id) : m_value(value), m_id(id) {}
    ValueRef value() const { return m_value; }
    size_t id() const { return m_id; }
};

class TracingValue final
        : public MixinValueImpl<TracingValue, ValueKind::Object, TracingInfo> {
public:
    using MixinValueImpl::MixinValueImpl;

    std::string to_string() const override {
        return ssprintf(
                "TracingValue{\"id\"=%zu, \"value\"=%s}", id(),
                value().to_string().c_str());
    }

    void on_watch() override { value().watch(); }

    void on_unwatch() override { value().unwatch(); }
};

/**
 * \brief trace operation sequence to TraceResult
 *
 * TracingTransformation records and forwards all operations to next layer,
 * as if it's transparent. When execution ends, it exports an operation sequence,
 * which is usually used to build CompiledTransformation.
 */
class TracingTransformation final : public Transformation {
public:
    using VarInfo = TraceResult::VarInfo;
    using VarKind = VarInfo::Kind;

private:
    std::vector<TraceResult::SeqItem> m_seq;
    std::vector<TraceResult::VarInfo> m_vars;
    std::vector<TracingValue::weak_ref_t> m_weak_vars;
    bool m_capture_as_const = false;
    bool m_record_input_shapes = false;

public:
    TracingTransformation(bool capture_as_const, bool record_input_shapes)
            : m_capture_as_const(capture_as_const),
              m_record_input_shapes(record_input_shapes) {}

    /**
     * \brief record values for trace
     *
     * \param value value to be traced
     * \param capture whether capture value or not
     * \param kind External, Constant or Internal
     * \return TypedValueRef<TracingValue> traced value
     */
    TypedValueRef<TracingValue> record_var(ValueRef value, bool capture, VarKind kind) {
        size_t id = m_vars.size();
        auto wrapped_value = TracingValue::make(value, id);
        m_vars.push_back({id, value.dtype(), value.device()});
        auto& var = m_vars.back();
        if (capture) {
            var.bound_data = value;
        }
        var.kind = kind;
        if (m_record_input_shapes && kind != VarKind::Internal) {
            var.shape = value.shape()->as_tensor_shape();
        }
        if (auto name = value.name()) {
            var.name = *name;
        }
        m_weak_vars.push_back(wrapped_value);
        return wrapped_value;
    }
    ValueRef unwrap_var(ValueRef value) {
        if (auto* tracing_value = value.as<TracingValue>()) {
            return tracing_value->value();
        }
        return value;
    }

    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        if (auto* tracing_value = value.as<TracingValue>()) {
            return tracing_value->value();
        }
        return value;
    }

    std::string name() const override { return "TracingTransformation"; }

    void on_unregister() noexcept override;

    TraceResult get_result() { return {m_seq, m_vars}; }
};

class TraceError : public std::exception {
private:
    std::string m_message;

public:
    TraceError(std::string reason) {
        m_message = ssprintf("trace error because %s", reason.c_str());
    }
    const char* what() const noexcept override { return m_message.c_str(); }
};

/**
 * \brief boost with traced result from TracingTransformation
 *
 * CompiledTransformation is built with an operation sequence. It compiles a megbrain
 * graph with the sequence and handle operation requests with this graph. Besides that,
 * it also checks that if current operation is same as previous one in seq.
 */
class CompiledTransformation final : public Transformation {
public:
    using VarInfo = TraceResult::VarInfo;
    using VarKind = VarInfo::Kind;

    struct VarAccessor {
        VarNode* node;
        std::function<TensorShape()> shape_getter;
        std::function<DeviceTensorND()> data_getter;
        std::function<HostTensorND()> value_getter;
        std::function<void(DeviceTensorND)> data_setter;
        std::function<void(std::exception_ptr)> exc_setter;
    };

    class TracedInfo {
    private:
        size_t m_id = 0;
        VarInfo* m_var = nullptr;
        VarAccessor* m_accessor = nullptr;
        mutable ShapeValue::ref_t m_shape;
        mutable DTypeValue::ref_t m_dtype;
        mutable CompNodeValue::ref_t m_comp_node;

    public:
        TracedInfo() = default;
        TracedInfo(size_t id, VarInfo* var, VarAccessor* accessor)
                : m_id(id), m_var(var), m_accessor(accessor) {}
        size_t id() const { return m_id; }
        ShapeValue::ref_t shape() const;
        DTypeValue::ref_t dtype() const;
        CompNodeValue::ref_t comp_node() const;
        const VarAccessor& accessor() const;

        void set_exception(std::exception_ptr exc) const {
            m_accessor->exc_setter(exc);
        }
    };

    class TracedValue final
            : public MixinValueImpl<TracedValue, ValueKind::Object, TracedInfo> {
    public:
        using MixinValueImpl::MixinValueImpl;

        std::string to_string() const override {
            return ssprintf("TracedValue{\"id\"=%zu}", id());
        }
    };

private:
    std::vector<TraceResult::SeqItem> m_seq;
    std::vector<TraceResult::VarInfo> m_vars;
    std::vector<VarAccessor> m_var_accessors;
    size_t m_pc = 0;
    std::shared_ptr<ComputingGraph> m_graph;
    std::unique_ptr<cg::AsyncExecutable> m_executable;
    std::vector<TracedValue::weak_ref_t> m_weak_values;
    std::thread m_graph_executor;
    std::function<bool(ValueRef, ValueRef)> m_value_comparator;
    bool m_input_shape_static;
    std::mutex m_mutex;
    std::exception_ptr m_graph_exc;
    std::vector<std::shared_ptr<BoxBase>> m_boxes;
    ComputingGraph::OutputSpec m_output_spec;

public:
    CompiledTransformation(TraceResult result, bool input_shape_static)
            : m_seq(result.seq),
              m_vars(result.vars),
              m_input_shape_static(input_shape_static) {
        m_graph = ComputingGraph::make();
        options().no_force_inplace = true;
        options().async_exec_level = 0b100;
    }

    ComputingGraph& graph() { return *m_graph; }

    ComputingGraph::Options& options() { return m_graph->options(); }

    /**
     * \brief Set the value comparator object (usually from python)
     *
     * \param comparator
     */
    void set_value_comparator(std::function<bool(ValueRef, ValueRef)> comparator) {
        m_value_comparator = comparator;
    }

    void compile();

    void recompile();

    void assert_tensor_equal(ValueRef lhs, ValueRef rhs);

    /**
     * \brief handle input for trace
     *
     * 1. For external, set input value to data_setter;
     * 2. For const, do nothing;
     * 3. For internal, assert var id;
     * *. Always assert data equals if there are data bound.
     *
     * \param id
     * \param value
     */
    void trace_input(size_t id, ValueRef value);

    /**
     * \brief make a placeholder for output.
     *
     * \param id trace_id
     * \return TracedValue::ref_t output placeholder, would be reset to real value when
     * trace exits
     */
    TracedValue::ref_t trace_output(size_t id);

    TraceResult::SeqItem& next_instruction();

    ValueRefList apply_op(const ApplyOp& apply_op, Span<ValueRef> inputs);

    ValueRefList apply_get_attr(const GetAttr& get_attr, Span<ValueRef> inputs);

    ValueRefList apply_create_tensor(
            const CreateTensor& create_tensor, Span<ValueRef> inputs);

    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    void on_unregister() noexcept override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is<TracedValue>());
        return value;
    }

    std::string name() const override { return "CompiledTransformation"; }

    void execute();

    void wait();

    std::exception_ptr set_exception(std::exception_ptr exc) noexcept;

    template <typename T>
    std::shared_ptr<Box<T>> make_box() {
        auto box = Box<T>::make();
        m_boxes.push_back(box);
        return box;
    }
};

}  // namespace mgb::imperative
