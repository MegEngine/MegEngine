#pragma once

#include <chrono>
#include <future>
#include <set>
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
        enum OpKind {
            Unknown,
            TraceMarkVar,
            Rename,
            IOMarkVar,
            CreateTensor,
        };
        std::shared_ptr<OpDef> op;
        SmallVector<size_t> inputs;
        SmallVector<size_t> outputs;
        OpKind kind = OpKind::Unknown;
    };

    using OpKind = SeqItem::OpKind;

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
        int handle_id;

        Kind kind;
        bool value_required = false;
        bool data_required = false;
        bool shape_required = false;
        std::set<size_t> inp_marker;
        std::set<size_t> out_marker;
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

    std::string raw_type() const { return "TraceMarkVar"; }
};

class IOMarkVar : public OperatorImpl<IOMarkVar, Operator::IdentityLike> {
public:
    enum Kind {
        Input,
        Output,
    };

private:
    size_t m_mark;
    Kind m_kind;

public:
    IOMarkVar(size_t mark, Kind kind) : m_mark(mark), m_kind(kind) {}

    size_t mark() const { return m_mark; }
    Kind kind() const { return m_kind; }

    std::string to_string() const override { return ssprintf("IOMarkVar"); }
    std::string raw_type() const override { return "IOMarkVar"; }
};

class TracingValue final : public ObjectValue<TracingValue> {
private:
    ValueRef m_value = {};
    size_t m_id = 0;

public:
    TracingValue(ValueRef value, size_t id) : m_value(value), m_id(id) {}
    ValueRef value() const { return m_value; }
    size_t id() const { return m_id; }

    std::string to_string() const override {
        return ssprintf(
                "TracingValue{\"id\"=%zu, \"value\"=%s}", id(),
                value().to_string().c_str());
    }

    void on_watch() override { value().watch(); }

    void on_unwatch() override { value().unwatch(); }

    void clear() override { m_value = {}; }
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
    using OpKind = TraceResult::SeqItem::OpKind;

private:
    std::vector<TraceResult::SeqItem> m_seq;
    std::vector<TraceResult::VarInfo> m_vars;
    std::vector<TracingValue::weak_ref_t> m_weak_vars;
    std::unordered_map<size_t, size_t> extern_var_to_id;
    bool m_capture_as_const = false;
    bool m_record_input_shapes = false;
    bool m_record_all_shapes = false;
    ObjectType<TracingValue> m_value_type{"TracingValue"};

public:
    std::unordered_map<size_t, size_t> inpmark_to_id;
    std::unordered_map<size_t, size_t> outmark_to_id;

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
        if (kind == VarKind::External &&
            extern_var_to_id.find(value.id()) != extern_var_to_id.end()) {
            return m_value_type.make(value, extern_var_to_id[value.id()]);
        }
        size_t id = m_vars.size();
        if (kind == VarKind::External) {
            extern_var_to_id[value.id()] = id;
        }
        auto wrapped_value = m_value_type.make(value, id);
        m_vars.push_back({id, value.dtype(), value.device()});
        auto& var = m_vars.back();
        if (capture) {
            var.bound_data = value;
        }
        var.kind = kind;
        if ((m_record_input_shapes && kind != VarKind::Internal) ||
            m_record_all_shapes) {
            var.shape = value.shape()->as_tensor_shape();
        }
        if (m_record_all_shapes)
            var.handle_id = value.handle_id();
        if (auto name = value.name()) {
            var.name = *name;
        }
        m_weak_vars.push_back(wrapped_value);
        return wrapped_value;
    }
    ValueRef unwrap_var(ValueRef value) {
        if (auto* tracing_value = value.as(m_value_type)) {
            return tracing_value->value();
        }
        return value;
    }

    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        if (auto* tracing_value = value.as(m_value_type)) {
            return tracing_value->value();
        }
        return value;
    }

    std::string name() const override { return "TracingTransformation"; }

    void on_unregister() noexcept override;
    void postprocess_trace_result();
    TraceResult get_result() { return {m_seq, m_vars}; }
    void enable_record_all_shapes() { m_record_all_shapes = true; }
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
    using OpKind = TraceResult::SeqItem::OpKind;

    struct VarAccessor {
        VarNode* node;  // use imperative mode when node == nullptr
        std::function<TensorShape()> shape_getter;
        std::function<DeviceTensorND()> data_getter;
        std::function<HostTensorND()> value_getter;
        std::function<void(DeviceTensorND)> data_setter;
        std::function<void(std::exception_ptr)> exc_setter;

        bool is_imperative() const { return node == nullptr; }
    };

    class TracedValue final : public ObjectValue<TracedValue> {
    private:
        size_t m_id = 0;
        VarInfo* m_var = nullptr;
        VarAccessor* m_accessor = nullptr;
        mutable ShapeValue::ref_t m_shape;
        mutable DTypeValue::ref_t m_dtype;
        mutable CompNodeValue::ref_t m_comp_node;
        mutable ValueRef m_imperative_value;

    public:
        TracedValue(size_t id, VarInfo* var, VarAccessor* accessor)
                : m_id(id), m_var(var), m_accessor(accessor) {}
        size_t id() const { return m_id; }
        ShapeValue::ref_t shape() const;
        DTypeValue::ref_t dtype() const;
        CompNodeValue::ref_t comp_node() const;
        DeviceValue::ref_t data() const;
        HostValue::ref_t value() const;
        const VarAccessor& accessor() const;

        void set_exception(std::exception_ptr exc) const {
            mgb_assert(m_accessor->exc_setter, "exc setter invalid");
            m_accessor->exc_setter(exc);
        }

        std::string to_string() const override {
            return ssprintf("TracedValue{\"id\"=%zu}", id());
        }

        void clear() override { m_imperative_value = {}; }

        void set_imperative_value(ValueRef value) const { m_imperative_value = value; }

        ValueRef get_imperative_value() const { return m_imperative_value; }
    };

private:
    std::vector<TraceResult::SeqItem> m_seq;
    std::vector<TraceResult::VarInfo> m_vars;
    std::vector<VarAccessor> m_var_accessors;
    std::unordered_map<std::string, size_t> mark2id;
    size_t m_pc = 0;
    std::shared_ptr<ComputingGraph> m_graph;
    std::unique_ptr<cg::AsyncExecutable> m_executable;
    std::vector<TracedValue::weak_ref_t> m_weak_values;
    std::thread m_graph_executor;
    std::function<bool(ValueRef, ValueRef)> m_value_comparator;
    bool m_input_shape_static;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::exception_ptr m_graph_exc;
    int m_graph_status = 0;  // 0 = stop, 1 = running, 2 = finalizing
    std::vector<std::shared_ptr<BoxBase>> m_boxes;
    ComputingGraph::OutputSpec m_output_spec;
    ObjectType<TracedValue> m_value_type{"TracedValue"};
    std::set<size_t> m_setted_extern;
    bool m_imperative = false;

public:
    CompiledTransformation(TraceResult result, bool input_shape_static, bool imperative)
            : m_seq(result.seq),
              m_vars(result.vars),
              m_input_shape_static(input_shape_static),
              m_imperative(imperative) {
        m_graph = ComputingGraph::make();
        options().no_force_inplace = true;
        options().async_exec_level = 0b100;
        if (!m_imperative) {
            start_worker();
        }
    }

    void start_worker() {
        m_graph_executor = std::thread([&] {
            while (true) {
                std::unique_lock lock{m_mutex};
                m_cv.wait(lock, [&] { return m_graph_status != 0; });
                lock.unlock();
                if (m_graph_status == 2) {
                    break;
                }
                try {
                    m_executable->execute();
                    m_executable->wait();
                } catch (...) {
                    auto exc = std::current_exception();
                    set_exception(exc);
                }
                lock.lock();
                m_graph_status = 0;
                lock.unlock();
                m_cv.notify_all();
            }
        });
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
    ValueRef trace_input(size_t id, ValueRef value);

    /**
     * \brief make a placeholder for output.
     *
     * \param id trace_id
     * \return TracedValue::ref_t output placeholder, would be reset to real value when
     * trace exits
     */
    TracedValue::ref_t trace_output(size_t id, ValueRef value);

    TraceResult::SeqItem& next_instruction();

    ValueRefList apply_op(const ApplyOp& apply_op, Span<ValueRef> inputs);

    ValueRefList apply_get_attr(const GetAttr& get_attr, Span<ValueRef> inputs);

    ValueRefList apply_create_tensor(
            const CreateTensor& create_tensor, Span<ValueRef> inputs);

    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    void on_unregister() noexcept override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is(m_value_type));
        return value;
    }

    VarAccessor& get_accessor_by_id(size_t id) { return m_var_accessors[id]; }

    std::string name() const override { return "CompiledTransformation"; }
    void set_pc_to_end() { m_pc = m_seq.size(); }
    void execute();

    void wait();

    void wait_worker();

    std::exception_ptr set_exception(std::exception_ptr exc) noexcept;

    template <typename T>
    std::shared_ptr<Box<T>> make_box() {
        auto box = Box<T>::make();
        m_boxes.push_back(box);
        return box;
    }

    void stop_worker() {
        {
            MGB_LOCK_GUARD(m_mutex);
            m_graph_status = 2;
        }
        m_cv.notify_all();
        m_graph_executor.join();
    }

    ~CompiledTransformation() {
        if (!m_imperative) {
            stop_worker();
        }
    }
};

}  // namespace mgb::imperative
