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

#include "megbrain/imperative/transformations/scalar.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/utility.h"

namespace mgb {
namespace imperative {

namespace {

using ScalarRule = std::function<std::vector<ValueRef>(const OpDef&, Span<ValueRef>)>;
static std::unordered_map<
        Typeinfo*, std::function<std::vector<ValueRef>(const OpDef&, Span<ValueRef>)>>
        scalar_rules;

ValueRef unwrap_input(ValueRef input) {
    if (auto scalar_input = input.as_ref<ScalarValue>()) {
        return scalar_input->value();
    } else {
        return input;
    }
}

std::vector<ValueRef> unwrap_inputs(Span<ValueRef> inputs) {
    std::vector<ValueRef> unwrapped_inputs;
    for (auto&& input : inputs) {
        unwrapped_inputs.push_back(unwrap_input(input));
    }
    return unwrapped_inputs;
}

ValueRef make_scalar_shape(CompNode device) {
    HostTensorND scalar_shape(device, {1}, dtype::Int32());
    scalar_shape.ptr<dt_int32>()[0] = 1;
    return imperative::apply(
            CreateTensor(CreateTensor::Const, device, scalar_shape.layout()),
            HostStorage::make(scalar_shape.storage()))[0];
}

bool is_scalar_shape(ValueRef shape) {
    if (shape.is<ScalarValue>()) {
        return false;
    }
    // may have performance issue
    auto shape_of_shape = shape.shape();
    if (!shape_of_shape) {
        // assume not scalar
        return false;
    }
    return *shape_of_shape == ValueShape{0};
}

template <typename T>
void register_scalar_rule(std::vector<ValueRef> (*rule)(const T&, Span<ValueRef>)) {
    scalar_rules[T::typeinfo()] = [rule](const OpDef& def, Span<ValueRef> inputs) {
        return (*rule)(def.cast_final_safe<T>(), inputs);
    };
}

std::vector<ValueRef> elemwise_rule(const Elemwise& elem, Span<ValueRef> inputs) {
    bool all_scalar = true;
    for (auto&& input : inputs) {
        if (!input.is<ScalarValue>()) {
            all_scalar = false;
            break;
        }
    }
    auto output = imperative::apply(elem, unwrap_inputs(inputs))[0];
    if (all_scalar) {
        return {ScalarValue::make(output)};
    } else {
        return {output};
    }
}

std::vector<ValueRef> remove_axis_rule(
        const RemoveAxis& remove_axis, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 1);
    mgb_assert(!inputs[0].is<ScalarValue>());
    auto output = imperative::apply(remove_axis, inputs)[0];
    bool is_scalar = inputs[0].shape()->ndim == remove_axis.axis.size();
    if (is_scalar) {
        return {ScalarValue::make(output)};
    } else {
        return {output};
    }
}

std::vector<ValueRef> reduce_rule(const Reduce& reduce, Span<ValueRef> inputs) {
    if (inputs.size() == 1) {
        return imperative::apply(reduce, unwrap_inputs(inputs));
    }
    mgb_assert(inputs.size() == 2);
    bool is_scalar = is_scalar_shape(inputs[1]);
    if (is_scalar) {
        auto unwrapped_input = unwrap_input(inputs[0]);
        CompNode device = *unwrapped_input.device();
        return {ScalarValue::make(imperative::apply(
                reduce, unwrapped_input, make_scalar_shape(device))[0])};
    }
    auto output = imperative::apply(reduce, unwrap_inputs(inputs))[0];
    if (is_scalar) {
        return {ScalarValue::make(output)};
    } else {
        return {output};
    }
}

std::vector<ValueRef> typecvt_rule(const TypeCvt& typecvt, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 1);
    if (auto scalar_input = inputs[0].as_ref<ScalarValue>()) {
        return {ScalarValue::make(
                imperative::apply(typecvt, scalar_input->value())[0])};
    } else {
        return imperative::apply(typecvt, inputs);
    }
}

std::vector<ValueRef> collective_comm_rule(
        const CollectiveComm& collective_comm, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 1);
    static std::unordered_set<CollectiveComm::Mode> modes = {
            CollectiveComm::Mode::ALL_REDUCE_MAX, CollectiveComm::Mode::ALL_REDUCE_MIN,
            CollectiveComm::Mode::ALL_REDUCE_SUM, CollectiveComm::Mode::BROADCAST,
            CollectiveComm::Mode::REDUCE_SUM,
    };
    if (modes.count(collective_comm.mode) == 0) {
        return imperative::apply(collective_comm, inputs);
    }
    if (auto scalar_input = inputs[0].as_ref<ScalarValue>()) {
        return {ScalarValue::make(
                imperative::apply(collective_comm, scalar_input->value())[0])};
    } else {
        return imperative::apply(collective_comm, inputs);
    }
}

std::vector<ValueRef> param_pack_split_rule(
        const ParamPackSplit& param_pack_split, Span<ValueRef> inputs) {
    auto outputs = imperative::apply(param_pack_split, unwrap_inputs(inputs));
    size_t nr_outputs = outputs.size();
    mgb_assert(nr_outputs == param_pack_split.shapes.size());
    for (size_t i = 0; i < nr_outputs; ++i) {
        if (param_pack_split.shapes[i].empty()) {
            outputs[i] = ScalarValue::make(outputs[i]);
        }
    }
    return outputs;
}

std::vector<ValueRef> dot_rule(const Dot& dot, Span<ValueRef> inputs) {
    return {ScalarValue::make(imperative::apply(dot, unwrap_inputs(inputs))[0])};
}

std::vector<ValueRef> add_axis_rule(const AddAxis& add_axis, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 1);
    if (auto scalar_input = inputs[0].as_ref<ScalarValue>()) {
        mgb_assert(add_axis.axis[0] == 0);
        if (add_axis.axis.size() == 1) {
            return {scalar_input->value()};
        } else {
            std::vector<int32_t> axis(add_axis.axis.begin() + 1, add_axis.axis.end());
            return imperative::apply(
                    ApplyOp(*AddAxis::make(axis, add_axis.scope())),
                    scalar_input->value());
        }
    } else {
        return imperative::apply(add_axis, inputs);
    }
}

std::vector<ValueRef> remote_recv_rule(
        const RemoteRecv& remote_recv, Span<ValueRef> inputs) {
    if (remote_recv.shape.empty()) {
        std::vector<int32_t> shape = {1};
        auto remote_recv_no_scalar = RemoteRecv::make(
                remote_recv.key, remote_recv.addr, remote_recv.port,
                remote_recv.rank_from, remote_recv.cn, shape, remote_recv.dtype,
                remote_recv.backend);
        remote_recv_no_scalar->set_scope(remote_recv.scope());
        return imperative::apply(
                ApplyOp(*remote_recv_no_scalar), unwrap_inputs(inputs));
    } else {
        return imperative::apply(remote_recv, unwrap_inputs(inputs));
    }
}

std::vector<ValueRef> check_no_finite_rule(
        const CheckNonFinite& check_no_finite, Span<ValueRef> inputs) {
    auto outputs = imperative::apply(check_no_finite, unwrap_inputs(inputs));
    mgb_assert(outputs.size() == inputs.size() + 1, "output size mismatch");
    outputs.back() = ScalarValue::make(outputs.back());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].is<ScalarValue>()) {
            outputs[i] = ScalarValue::make(outputs[i]);
        }
    }
    return outputs;
}

std::vector<ValueRef> subtensor_rule(
        const Subtensor& subtensor, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() >= 1);
    auto input = inputs[0];
    bool is_scalar;
    mgb_assert(!input.is<ScalarValue>(), "subtensor shouldn't have scalar input");
    if (auto shape = input.shape()) {
        size_t ndim = input.shape()->ndim;
        for (auto&& [axis, begin, end, step, idx] : subtensor.items) {
            if (idx) {
                ndim--;
            }
        }
        is_scalar = ndim == 0;
    } else {
        is_scalar = false;
    }
    auto output = imperative::apply(subtensor, unwrap_inputs(inputs))[0];
    if (is_scalar) {
        return {ScalarValue::make(output)};
    } else {
        return {output};
    }
}

std::vector<ValueRef> get_var_shape_rule(
        const GetVarShape& get_var_shape, Span<ValueRef> inputs) {
    bool all_scalar = true;
    mgb_assert(inputs.size() >= 1);
    for (auto&& input : inputs) {
        if (!input.is<ScalarValue>()) {
            all_scalar = false;
        }
    }
    if (all_scalar) {
        auto device = inputs[0].cast<ScalarValue>().value().device();
        auto storage = HostStorage::make(*device);
        // storage->ensure_size(1);
        return imperative::apply(
                CreateTensor(
                        CreateTensor::Const, *device, dtype::Int32(), ValueShape{0}),
                storage);
    } else {
        return imperative::apply(get_var_shape, unwrap_inputs(inputs));
    }
}

std::vector<ValueRef> fastpath_copy_rule(
        const FastpathCopy& fastpath_copy, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 1);
    bool is_scalar = inputs[0].is<ScalarValue>();
    auto output = imperative::apply(fastpath_copy, unwrap_inputs(inputs))[0];
    if (is_scalar) {
        return {ScalarValue::make(output)};
    } else {
        return {output};
    }
}

std::vector<ValueRef> reshape_rule(const Reshape& reshape, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 2);
    bool is_scalar = is_scalar_shape(inputs[1]);
    auto unwrapped_input = inputs[0].is<ScalarValue>()
                                 ? inputs[0].cast<ScalarValue>().value()
                                 : inputs[0];
    if (is_scalar) {
        return {ScalarValue::make(imperative::apply(
                reshape, unwrapped_input,
                make_scalar_shape(*unwrapped_input.device()))[0])};
    } else {
        return imperative::apply(reshape, unwrap_inputs(inputs));
    }
}

std::vector<ValueRef> broadcast_rule(
        const Broadcast& broadcast, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 2);
    bool is_scalar = is_scalar_shape(inputs[1]);
    auto unwrapped_input = inputs[0].is<ScalarValue>()
                                 ? inputs[0].cast<ScalarValue>().value()
                                 : inputs[0];
    if (is_scalar) {
        return {ScalarValue::make(imperative::apply(
                broadcast, unwrapped_input,
                make_scalar_shape(*unwrapped_input.device()))[0])};
    } else {
        return imperative::apply(broadcast, unwrap_inputs(inputs));
    }
}

std::vector<ValueRef> copy_rule(const Copy& copy, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 1);
    bool is_scalar = inputs[0].is<ScalarValue>();
    if (is_scalar) {
        return {ScalarValue::make(imperative::apply(copy, unwrap_inputs(inputs))[0])};
    } else {
        return imperative::apply(copy, unwrap_inputs(inputs));
    }
}

std::vector<ValueRef> inplace_add_rule(
        const InplaceAdd& inplace_add, Span<ValueRef> inputs) {
    mgb_assert(inputs.size() == 4);
    bool is_scalar = inputs[0].is<ScalarValue>();
    if (is_scalar) {
        return {ScalarValue::make(
                imperative::apply(inplace_add, unwrap_inputs(inputs))[0])};
    } else {
        return imperative::apply(inplace_add, unwrap_inputs(inputs));
    }
}

template <typename T>
std::vector<ValueRef> subgraph_op_rule(const T& op, Span<ValueRef> inputs) {
    // TODO: add flag instead of assume
    bool all_scalar = true;
    for (auto&& input : inputs) {
        if (!input.is<ScalarValue>()) {
            all_scalar = false;
        }
    }
    auto outputs = imperative::apply(op, unwrap_inputs(inputs));
    if (all_scalar) {
        for (auto& output : outputs) {
            output = ScalarValue::make(output);
        }
    }
    return outputs;
}

struct ScalarRuleRegistry {
    ScalarRuleRegistry() {
        register_scalar_rule(elemwise_rule);
        register_scalar_rule(remove_axis_rule);
        register_scalar_rule(reduce_rule);
        register_scalar_rule(typecvt_rule);
        register_scalar_rule(collective_comm_rule);
        register_scalar_rule(param_pack_split_rule);
        register_scalar_rule(dot_rule);
        register_scalar_rule(add_axis_rule);
        register_scalar_rule(remote_recv_rule);
        register_scalar_rule(check_no_finite_rule);
        register_scalar_rule(subtensor_rule);
        register_scalar_rule(get_var_shape_rule);
        register_scalar_rule(fastpath_copy_rule);
        register_scalar_rule(reshape_rule);
        register_scalar_rule(broadcast_rule);
        register_scalar_rule(copy_rule);
        register_scalar_rule(inplace_add_rule);
        register_scalar_rule(subgraph_op_rule<SubgraphOp>);
        register_scalar_rule(subgraph_op_rule<CompiledOp>);
    }
} _;
}  // namespace

std::vector<ValueRef> ScalarTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto apply_op = op.as<ApplyOp>()) {
        auto iter = scalar_rules.find(apply_op->op().dyn_typeinfo());
        if (iter != scalar_rules.end()) {
            return iter->second(apply_op->op(), inputs);
        } else {
            // TODO: repeat op
            return imperative::apply(op, unwrap_inputs(inputs));
        }
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        if (create_tensor->shape().is_scalar()) {
            ValueShape scalar_shape = {1};
            CreateTensor scalar_op(
                    create_tensor->kind(), create_tensor->device(),
                    create_tensor->dtype(), scalar_shape);
            return {ScalarValue::make(imperative::apply(scalar_op, inputs)[0])};
        } else {
            return imperative::apply(op, inputs);
        }
    } else if (auto* get_attr = op.as<GetAttr>()) {
        bool is_scalar = inputs.as_array<1>()[0].is<ScalarValue>();
        auto output = imperative::apply(op, unwrap_inputs(inputs))[0];
        if (!is_scalar) {
            return {output};
        }
        switch (get_attr->attr()) {
            case GetAttr::Shape: {
                // Scalar Shape
                return {ShapeValue::make()};
            }
            case GetAttr::Value: {
                auto& hv = output.cast<HostValue>();
                mgb_assert(
                        hv.shape() == ValueShape({1}),
                        "underlying value should has shape {1}, got %s",
                        hv.shape().to_string().c_str());
                return {HostValue::make(hv.dtype(), ValueShape(), hv.storage())};
            }
            case GetAttr::Data: {
                auto& dv = output.cast<DeviceValue>();
                mgb_assert(
                        dv.shape() == ValueShape({1}),
                        "underlying value should has shape {1}, got %s",
                        dv.shape().to_string().c_str());
                return {DeviceValue::make(dv.dtype(), ValueShape(), dv.storage())};
            }
            default:
                return {output};
        }
    } else if (op.as<IsScalar>()) {
        return {BoolValue::make(inputs.as_array<1>()[0].is<ScalarValue>())};
    } else if (op.is<Operator::IdentityLike>()) {
        bool is_scalar = inputs.as_array<1>()[0].is<ScalarValue>();
        if (is_scalar) {
            return {ScalarValue::make(imperative::apply(op, unwrap_inputs(inputs))[0])};
        } else {
            return imperative::apply(op, inputs);
        }
    } else {
        return imperative::apply(op, unwrap_inputs(inputs));
    }
};

}  // namespace imperative
}  // namespace mgb
