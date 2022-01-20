/**
 * \file imperative/src/impl/transformations/trace.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/transformations/trace.h"

#include <chrono>
#include <exception>

#include "megbrain/gopt/inference.h"
#include "megbrain/graph/helper.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/serializer.h"

#include "../event_pool.h"

#define trace_assert(_cond, _msg...)                                        \
    do {                                                                    \
        if (mgb_unlikely(!(_cond))) {                                       \
            auto exc = std::make_exception_ptr(TraceError(ssprintf(_msg))); \
            set_exception(exc);                                             \
            std::rethrow_exception(exc);                                    \
        }                                                                   \
    } while (0)

namespace mgb {
namespace imperative {

VarNodeArray TraceResult::dump(
        ComputingGraph& graph,
        std::vector<std::tuple<size_t, std::string, TensorShape>> inputs,
        std::vector<std::pair<size_t, std::string>> outputs, bool prefer_input_names) {
    // var -> VarNode
    std::vector<VarNode*> nodes(vars.size(), nullptr);
    // make h2d node for each input
    for (auto&& [input, name, shape] : inputs) {
        auto& var = vars[input];
        auto& node = nodes[input];
        // TODO: cambricon CompNode
        auto host = std::make_shared<HostTensorND>(
                CompNode::load("xpux"), shape, var.dtype);
        OperatorNodeConfig config;
        // if prefer_input_names, prefer names from dump args
        // else prefer names got from trace procedure
        if (prefer_input_names && !name.empty()) {
            config.name(name);
        } else if (!var.name.empty()) {
            config.name(var.name);
        } else if (!name.empty()) {
            config.name(name);
        }
        node = opr::Host2DeviceCopy::make(graph, host, {}, config).node();
    }
    // make const node for each constant
    for (size_t i = 0; i < vars.size(); ++i) {
        auto& var = vars[i];
        auto& node = nodes[i];
        if (!node) {
            if (var.kind != VarKind::Internal) {
                if (!var.bound_data) {
                    continue;
                }
                if (!var.name.empty()) {
                    node = opr::ImmutableTensor::make(
                                   graph, var.bound_data.numpy()->as_nd(), {var.name})
                                   .node();
                } else {
                    node = opr::ImmutableTensor::make(
                                   graph, var.bound_data.numpy()->as_nd())
                                   .node();
                }
            }
        }
    }
    std::unordered_map<std::string, std::vector<cg::OperatorNodeBase*>> name2ops;
    // iterate over opr_seq
    for (auto&& item : seq) {
        auto&& [op, inputs, outputs] = item;
        VarNodeArray input_nodes;
        for (auto&& input : inputs) {
            auto& node = nodes[input];
            input_nodes.push_back(node);
        }
        VarNodeArray output_nodes;
        if (op) {
            if (auto* bn = op->try_cast_final<BatchNorm>()) {
                mgb_assert(
                        bn->fwd_mode == BatchNorm::FwdMode::INFERENCE,
                        "can not dump BatchNorm in training mode, maybe you forget to "
                        "do model.eval()?");
            }
            output_nodes = OpDef::apply_on_var_node(*op, input_nodes);
            name2ops[output_nodes[0]->owner_opr()->name()].push_back(
                    output_nodes[0]->owner_opr());
        } else {
            // no opr, just forward VarNode
            mgb_assert(
                    inputs.size() == outputs.size(),
                    "output size not equals to input size when forwarding");
            output_nodes = input_nodes;
        }
        mgb_assert(output_nodes.size() == outputs.size(), "output size mismatch");
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto output = outputs[i];
            auto& var = vars[output];
            auto& node = nodes[output];
            mgb_assert(var.kind == VarKind::Internal, "output node should be internal");
            if (!node) {
                node = output_nodes[i];
            }
            if (!var.name.empty()) {
                node->name(var.name);
            }
        }
    }
    for (auto&& [name, ops] : name2ops) {
        if (ops.size() <= 1) {
            continue;
        }
        // ops.size() > 1, need dedup (rename op)
        for (size_t i = 0; i < ops.size(); ++i) {
            auto& op = ops[i];
            auto new_name = ssprintf("%s[%zu]", name.c_str(), i);
            for (auto&& output : op->output()) {
                auto output_name = output->name();
                auto pos = output_name.find(name);
                if (pos != std::string::npos) {
                    output_name.replace(pos, name.length(), new_name);
                }
                output->name(output_name);
            }
            op->name(new_name);
        }
    }
    VarNodeArray output_nodes;
    for (auto&& [output, name] : outputs) {
        mgb_assert(output < vars.size(), "invalid output id %zu", output);
        mgb_assert(nodes[output], "output node invalid");
        if (!name.empty()) {
            nodes[output]->name(name);
        }
        output_nodes.push_back(nodes[output]);
    }
    return output_nodes;
}

ValueRefList TracingTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* op_value = op.as<ApplyOp>()) {
        SmallVector<ValueRef> unwrapped_inputs;
        SmallVector<TracingValue::ref_t> wrapped_inputs;
        SmallVector<size_t> input_ids;
        for (auto input : inputs) {
            auto tracing_value = input.as_ref<TracingValue>();
            if (!tracing_value) {
                tracing_value =
                        record_var(input, m_capture_as_const, VarKind::External);
            }
            unwrapped_inputs.push_back(tracing_value->value());
            wrapped_inputs.push_back(tracing_value);
            input_ids.push_back(tracing_value->id());
        }
        // TODO: remove OpDef::set_scope
        auto scopes = Transformation::scopes();
        std::string scopes_join;
        for (auto&& scope : scopes) {
            if (!scopes_join.empty()) {
                scopes_join.push_back('.');
            }
            scopes_join.append(scope);
        }
        const_cast<OpDef&>(op_value->op()).set_scope(scopes_join);
        auto unwrapped_outputs = imperative::apply(op, unwrapped_inputs);
        ValueRefList wrapped_outputs(unwrapped_outputs.size());
        SmallVector<size_t> output_ids;
        for (size_t i = 0; i < unwrapped_outputs.size(); ++i) {
            auto&& output = unwrapped_outputs[i];
            auto wrapped_output = record_var(output, false, VarKind::Internal);
            wrapped_outputs[i] = wrapped_output;
            output_ids.push_back(wrapped_output->id());
        }
        m_seq.push_back({op_value->op().shared_from_this(), input_ids, output_ids});
        return wrapped_outputs;
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        auto outputs = imperative::apply(op, inputs);
        if (create_tensor->kind() == CreateTensor::NoTrace) {
            return outputs;
        }
        bool is_const = create_tensor->kind() == CreateTensor::Const;
        auto wrapped_input = record_var(
                outputs[0], is_const || m_capture_as_const,
                is_const ? VarKind::Constant : VarKind::External);
        auto wrapped_output = record_var(outputs[0], false, VarKind::Internal);
        auto input_id = wrapped_input->id();
        auto output_id = wrapped_output->id();
        m_seq.push_back({{}, {input_id}, {output_id}});
        return {wrapped_output};
    } else if (auto* get_attr = op.as<GetAttr>()) {
        auto unwrapped_input = unwrap_var(inputs[0]);
        auto outputs = imperative::apply(op, unwrapped_input);
        if (auto* tracing_value = inputs[0].as<TracingValue>()) {
            auto& var_info = m_vars[tracing_value->id()];
            switch (get_attr->attr()) {
                case GetAttr::Shape:
                    // TODO: reduce h2d when data or value is available
                    var_info.shape_required = true;
                    break;
                case GetAttr::Data:
                    var_info.data_required = true;
                    break;
                case GetAttr::Value:
                    var_info.value_required = true;
                    break;
                default:
                    break;
            }
        }
        return outputs;
    } else if (auto* trace_mark_var = op.as<TraceMarkVar>()) {
        mgb_assert(inputs.size() == 1, "TraceMarkVar expects exactly one input");
        auto input = inputs[0];
        auto tracing_var = input.as_ref<TracingValue>();
        if (!tracing_var) {
            bool is_input = trace_mark_var->mark().substr(0, 4) == "arg_" ||
                            trace_mark_var->mark().substr(0, 6) == "kwarg_";
            if (is_input) {
                tracing_var = record_var(input, false, VarKind::External);
            } else {
                tracing_var = record_var(input, m_capture_as_const, VarKind::External);
            }
        } else {
            input = tracing_var->value();
        }
        auto output = record_var(input, false, VarKind::Internal);
        m_vars[output->id()].mark = trace_mark_var->mark();
        m_seq.push_back({{}, {tracing_var->id()}, {output->id()}});
        return {output};
    } else if (auto* trace_name_var = op.as<RenameValue>()) {
        mgb_assert(inputs.size() == 1, "RenameValue expects exactly one input");
        auto input = inputs[0];
        auto tracing_var = input.as_ref<TracingValue>();
        if (!tracing_var) {
            tracing_var = record_var(input, m_capture_as_const, VarKind::External);
        } else {
            input = tracing_var->value();
        }
        auto output = record_var(input, false, VarKind::Internal);
        m_vars[output->id()].name = trace_name_var->name();
        m_seq.push_back({{}, {tracing_var->id()}, {output->id()}});
        return {output};
    } else if (op.is<GetName>()) {
        mgb_assert(inputs.size() == 1, "GetName expects exactly one input");
        auto input = inputs[0];
        if (auto tracing_var = input.as_ref<TracingValue>()) {
            auto name = m_vars[tracing_var->id()].name;
            if (!name.empty()) {
                return {StringValue::make(name)};
            } else {
                return {ValueRef()};
            }
        }
        return imperative::apply(op, inputs);
    } else {
        // TODO: handle DTRCommand and ...
        return op.fallback(inputs);
    }
}

void TracingTransformation::on_unregister() noexcept {
    for (auto&& weak_var : m_weak_vars) {
        if (auto tracing_value = weak_var.lock()) {
            auto& var_info = m_vars[tracing_value->id()];
            var_info.data_required = true;
            tracing_value.reset(tracing_value->value());
        }
    }
    m_weak_vars.clear();
}

void CompiledTransformation::compile() {
    // these ops require seq order, so we link them to an mm_io_link to ensure order
    static std::unordered_set<Typeinfo*> mm_io_ops = {
            CollectiveComm::typeinfo(), RemoteSend::typeinfo(), RemoteRecv::typeinfo()};
    mgb_assert(!m_executable, "already compiled");
    // FIXME: mm_io_link and io_links should be merged
    SymbolVarArray io_links;
    SymbolVar mm_io_link;
    auto make_input = [&](VarInfo* var_info) {
        mgb_assert(
                var_info->kind == VarKind::External, "input node should be external");
        VarAccessor accessor;
        auto box = make_box<DeviceTensorND>();
        // TODO: attach ref count, release early
        auto outputs = opr::InputCallback::make(
                *m_graph, [box] { return box->take_value(); }, var_info->device,
                var_info->dtype, var_info->shape, io_links, m_input_shape_static);
        // attach input_callback to io_links
        accessor.node = outputs[0].node();
        io_links = {outputs[1]};
        accessor.data_setter = [box](DeviceTensorND data) { box->try_set_value(data); };
        return accessor;
    };
    auto make_output = [&](TraceResult::VarInfo* var_info, SymbolVar node) {
        VarAccessor accessor;
        accessor.node = node.node();
        if (var_info->shape_required) {
            // TODO: use static infer manager for some vars?
            auto box = make_box<TensorShape>();
            auto callback = [box](DeviceTensorND data) {
                box->try_set_value(data.shape());
            };
            SymbolVarArray inputs = io_links;
            inputs.insert(inputs.begin(), node);
            auto output = opr::OutputCallback::make({callback, true, false}, inputs);
            io_links = {output};
            accessor.shape_getter = [box]() -> TensorShape { return box->get_value(); };
        }
        if (var_info->data_required) {
            auto box = make_box<DeviceTensorND>();
            auto callback = [box](DeviceTensorND data) { box->try_set_value(data); };
            SymbolVarArray inputs = io_links;
            inputs.insert(inputs.begin(), node);
            auto output = opr::OutputCallback::make({callback, false, false}, inputs);
            io_links = {output};
            accessor.data_getter = [box]() -> DeviceTensorND {
                return box->get_value();
            };
        }
        if (var_info->value_required) {
            struct ValueWithEvent {
                HostTensorND value;
                CompNode::Event* event = nullptr;
            };
            auto box = make_box<ValueWithEvent>();
            auto event = EventPool::without_timer().alloc_shared(var_info->device);
            auto callback = [box, event](DeviceTensorND data) {
                HostTensorND host_val;
                host_val.copy_from(data);
                if (data.comp_node() != CompNode::default_cpu()) {
                    mgb_assert(data.comp_node() == event->comp_node());
                    event->record();
                    box->try_set_value({host_val, event.get()});
                } else {
                    box->try_set_value({host_val});
                }
            };
            SymbolVarArray inputs = io_links;
            inputs.insert(inputs.begin(), node);
            auto output = opr::OutputCallback::make({callback, false, true}, inputs);
            io_links = {output};
            accessor.value_getter = [box]() -> HostTensorND {
                auto&& [value, event] = box->get_value();
                if (event) {
                    event->host_wait();
                }
                return value;
            };
        }
        return accessor;
    };
    auto make_const = [&](TraceResult::VarInfo* var_info) {
        VarAccessor accessor;
        mgb_assert(
                var_info->kind == VarKind::Constant, "const node should be constant");
        HostTensorND host_val = var_info->bound_data.numpy()->as_nd();
        accessor.node = opr::ImmutableTensor::make(*m_graph, host_val).node();
        return accessor;
    };
    std::vector<VarAccessor> var_accessors(m_vars.size());
    auto exc_setter = std::bind(
            &CompiledTransformation::set_exception, this, std::placeholders::_1);
    for (auto&& accessor : var_accessors) {
        accessor.exc_setter = exc_setter;
    }
    for (auto&& item : m_seq) {
        bool require_link = bool(item.op) && mm_io_ops.count(item.op->dyn_typeinfo());
        VarNodeArray input_vars;
        for (auto&& input : item.inputs) {
            auto& var = m_vars[input];
            if (!var_accessors[input].node) {
                switch (var.kind) {
                    case VarKind::External:
                        var_accessors[input] = make_input(&var);
                        break;
                    case VarKind::Constant:
                        var_accessors[input] = make_const(&var);
                        break;
                    default:
                        mgb_throw(
                                AssertionError,
                                "internal node should be valid when used as input");
                }
            }
            input_vars.push_back(var_accessors[input].node);
        }
        if (require_link && mm_io_link.node()) {
            mgb_assert(
                    !input_vars.empty(),
                    "io-mm operator should have at least one input");
            input_vars[0] =
                    opr::VirtualDep::make({SymbolVar(input_vars[0]), mm_io_link})
                            .node();
        }
        VarNodeArray output_vars;
        if (item.op) {
            output_vars = OpDef::apply_on_var_node(*item.op, input_vars);
        } else {
            // forward inputs to outputs
            mgb_assert(
                    item.inputs.size() == item.outputs.size(),
                    "output size not equals to input size when forwarding");
            for (auto&& input_var : input_vars) {
                output_vars.push_back(input_var);
            }
        }
        if (require_link) {
            mgb_assert(
                    !item.outputs.empty(),
                    "io-mm operator should have at least one output");
            mm_io_link = SymbolVar(output_vars[0]);
        }
        // init output accessors
        for (size_t i = 0; i < output_vars.size(); ++i) {
            auto output = item.outputs[i];
            auto& node = output_vars[i];
            auto& var = m_vars[output];
            var_accessors[output] = make_output(&var, node);
        }
    }
    ComputingGraph::OutputSpec output_specs;
    // avoid input/output/callback from being optimized
    for (auto&& io_link : io_links) {
        output_specs.push_back({io_link, {}});
    }
    // avoid remote io ops from being optimized
    if (mm_io_link.node()) {
        output_specs.push_back({mm_io_link, {}});
    }
    {
        // set_priority_to_id
        // workaround for having mm_io_link and io_links separated
        auto on_opr = [](mgb::cg::OperatorNodeBase* opr) {
            if (opr->node_prop().attribute().priority == 0) {
                opr->node_prop().attribute().priority = opr->id();
            }
        };
        mgb::cg::DepOprIter dep_iter{on_opr};
        for (const auto& output_spec : output_specs) {
            dep_iter.add(output_spec.first);
        }
    }
    m_executable = m_graph->compile(output_specs);
    m_var_accessors = var_accessors;
    m_output_spec = output_specs;
}

void CompiledTransformation::recompile() {
    mgb_assert(m_executable);
    m_executable = m_graph->compile(m_output_spec);
}

void CompiledTransformation::assert_tensor_equal(ValueRef lhs, ValueRef rhs) {
    trace_assert(m_value_comparator(lhs, rhs), "tensors not equals");
}

void CompiledTransformation::trace_input(size_t id, ValueRef value) {
    try {
        auto& var = m_vars[id];
        auto& var_accessor = m_var_accessors[id];
        switch (var.kind) {
            case VarKind::External: {
                trace_assert(
                        !value.is<TracedValue>(), "expect external node, got internal");
                if (var.bound_data) {
                    assert_tensor_equal(var.bound_data, value);
                } else {
                    DType dtype = *value.dtype();
                    CompNode device = *value.device();
                    trace_assert(
                            var.dtype == dtype, "dtype mismatch: %s vs %s",
                            var.dtype.name(), dtype.name());
                    trace_assert(
                            var.device == device, "comp_node mismatch: %s vs %s",
                            var.device.to_string().c_str(), device.to_string().c_str());
                }
                var_accessor.data_setter(value.dev_tensor()->as_nd());
                break;
            }
            case VarKind::Constant: {
                mgb_assert(var.bound_data, "const var without data bound");
                assert_tensor_equal(var.bound_data, value);
                break;
            }
            case VarKind::Internal: {
                trace_assert(
                        value.is<TracedValue>(), "expect internal node, got external");
                auto& traced_value = value.cast<TracedValue>();
                trace_assert(traced_value.id() == id, "input id mismatch");
                break;
            }
        }
    } catch (TraceError&) {
        throw;
    } catch (...) {
        mgb_assert(false, "unexpected error");
    }
}

auto CompiledTransformation::trace_output(size_t id) -> TracedValue::ref_t {
    auto traced_value = TracedValue::make(id, &m_vars[id], &m_var_accessors[id]);
    m_weak_values.push_back(traced_value);
    return traced_value;
}

TraceResult::SeqItem& CompiledTransformation::next_instruction() {
    trace_assert(m_pc < m_seq.size(), "too many instructions");
    return m_seq[m_pc++];
}

ShapeValue::ref_t CompiledTransformation::TracedInfo::shape() const {
    if (!m_shape) {
        trace_assert(m_accessor->shape_getter, "shape unreadable");
        m_shape = ShapeValue::make(ValueShape::from(m_accessor->shape_getter()));
    }
    return m_shape;
}

DTypeValue::ref_t CompiledTransformation::TracedInfo::dtype() const {
    if (!m_dtype) {
        m_dtype = DTypeValue::make(m_var->dtype);
    }
    return m_dtype;
}

CompNodeValue::ref_t CompiledTransformation::TracedInfo::comp_node() const {
    if (!m_comp_node) {
        m_comp_node = CompNodeValue::make(m_var->device);
    }
    return m_comp_node;
}
auto CompiledTransformation::TracedInfo::accessor() const -> const VarAccessor& {
    return *m_accessor;
}

ValueRefList CompiledTransformation::apply_op(
        const ApplyOp& apply_op, Span<ValueRef> inputs) {
    auto& item = next_instruction();
    trace_assert(inputs.size() == item.inputs.size(), "input size mismatch");
    trace_assert(apply_op.op().is_same(*item.op), "operator mismatch");
    for (size_t i = 0; i < inputs.size(); ++i) {
        trace_input(item.inputs[i], inputs[i]);
    }
    ValueRefList outputs(item.outputs.size());
    for (size_t i = 0; i < item.outputs.size(); ++i) {
        outputs[i] = trace_output(item.outputs[i]);
    }
    return outputs;
}

ValueRefList CompiledTransformation::apply_get_attr(
        const GetAttr& get_attr, Span<ValueRef> inputs) {
    if (auto* traced_value = inputs[0].as<TracedValue>()) {
        ValueRef output;
        auto& var_accessor = traced_value->accessor();
        switch (get_attr.attr()) {
            case GetAttr::Shape:
                output = traced_value->shape();
                break;
            case GetAttr::Data:
                trace_assert(var_accessor.data_getter, "data unreadable");
                output = DeviceValue::make(var_accessor.data_getter());
                break;
            case GetAttr::Value:
                trace_assert(var_accessor.value_getter, "value unreadable");
                output = HostValue::make(var_accessor.value_getter());
                break;
            case GetAttr::DType:
                output = traced_value->dtype();
                break;
            case GetAttr::Device:
                output = traced_value->comp_node();
            default:
                break;
        }
        return {output};
    } else {
        return imperative::apply(get_attr, inputs);
    }
}

ValueRefList CompiledTransformation::apply_create_tensor(
        const CreateTensor& create_tensor, Span<ValueRef> inputs) {
    if (create_tensor.kind() == CreateTensor::NoTrace) {
        return imperative::apply(create_tensor, inputs);
    }
    auto& item = next_instruction();
    trace_assert(item.op == nullptr, "operator mismatch");
    auto input_id = item.inputs[0];
    auto output_id = item.outputs[0];
    auto tensor = imperative::apply(create_tensor, inputs)[0];
    trace_input(input_id, tensor);
    return {trace_output(output_id)};
}

ValueRefList CompiledTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* op_value = op.as<ApplyOp>()) {
        return apply_op(*op_value, inputs);
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        return apply_create_tensor(*create_tensor, inputs);
    } else if (auto* get_attr = op.as<GetAttr>()) {
        return apply_get_attr(*get_attr, inputs);
    } else if (auto* trace_mark_var = op.as<TraceMarkVar>()) {
        auto& item = next_instruction();
        trace_assert(item.op == nullptr, "operator mismatch");
        trace_assert(item.inputs.size() == 1, "inputs size mismatch");
        trace_assert(item.outputs.size() == 1, "inputs output mismatch");
        trace_input(item.inputs[0], inputs[0]);
        trace_assert(
                trace_mark_var->mark() == m_vars[item.outputs[0]].mark,
                "mark mismatch");
        return {trace_output(item.outputs[0])};
    } else if (auto* trace_name_var = op.as<RenameValue>()) {
        auto& item = next_instruction();
        trace_assert(item.op == nullptr, "operator mismatch");
        trace_assert(item.inputs.size() == 1, "inputs size mismatch");
        trace_assert(item.outputs.size() == 1, "outputs size mismatch");
        trace_input(item.inputs[0], inputs[0]);
        trace_assert(
                trace_name_var->name() == m_vars[item.outputs[0]].name,
                "name mismatch");
        return {trace_output(item.outputs[0])};
    } else {
        return op.fallback(inputs);
    }
}

void CompiledTransformation::on_unregister() noexcept {
    // resolve pending values
    for (auto&& weak_value : m_weak_values) {
        if (auto traced_value = weak_value.lock()) {
            auto& var_accessor = m_var_accessors[traced_value->id()];
            auto value = ([&]() -> ValueRef {
                try {
                    trace_assert(var_accessor.data_getter, "data unreadable");
                    auto dev_value = DeviceValue::make(var_accessor.data_getter());
                    return imperative::apply(
                            CreateTensor(
                                    CreateTensor::Common, dev_value->device(),
                                    dev_value->dtype(), dev_value->shape()),
                            DeviceStorage::make(dev_value->storage()))[0];
                } catch (...) {
                    set_exception(std::current_exception());
                    return ErrorValue::make("trace exit failed");
                }
            })();
            traced_value.reset(value);
        }
    }
    m_weak_values.clear();
}

void CompiledTransformation::execute() {
    mgb_assert(m_executable != nullptr);
    m_graph_executor = std::thread([&] {
        try {
            m_executable->execute();
            m_executable->wait();
        } catch (...) {
            auto exc = std::current_exception();
            set_exception(exc);
        }
    });
}

void CompiledTransformation::wait() {
    try {
        trace_assert(m_pc == m_seq.size(), "mismature end");
    } catch (...) {
    }
    mgb_assert(m_executable != nullptr);
    m_graph_executor.join();
    m_graph_executor = {};
    for (auto&& box : m_boxes) {
        box->reset();
    }
    m_pc = 0;
    std::exception_ptr graph_exc;
    std::swap(m_graph_exc, graph_exc);
    if (graph_exc) {
        // graph with exception cannot be reused
        recompile();
        std::rethrow_exception(graph_exc);
    }
}

std::exception_ptr CompiledTransformation::set_exception(
        std::exception_ptr exc) noexcept {
    MGB_LOCK_GUARD(m_mutex);
    if (m_graph_exc) {
        return m_graph_exc;
    }
    for (auto&& box : m_boxes) {
        box->try_set_exception(exc);
    }
    m_graph_exc = exc;
    return m_graph_exc;
}

}  // namespace imperative
}  // namespace mgb
