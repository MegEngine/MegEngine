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
#include "megbrain/imperative/utils/stats.h"

namespace mgb {
namespace imperative {

namespace {

using ScalarRule = ValueRefList (*)(
        const OpDef&, Span<ValueRef>, Span<bool>, const Type<ScalarValue>&);
static std::unordered_map<Typeinfo*, ScalarRule> scalar_rules;

ValueRef make_scalar_shape(CompNode device) {
    HostTensorND scalar_shape(device, {1}, dtype::Int32());
    scalar_shape.ptr<dt_int32>()[0] = 1;
    return imperative::apply(
            CreateTensor(CreateTensor::Const, device, scalar_shape.layout()),
            HostStorage::make(scalar_shape.storage()))[0];
}

bool is_scalar_shape(ValueRef shape) {
    // may have performance issue
    auto shape_of_shape = shape.shape();
    if (!shape_of_shape) {
        // assume not scalar
        return false;
    }
    return *shape_of_shape == ValueShape{0};
}

template <
        typename T,
        ValueRefList (*rule)(
                const T&, Span<ValueRef>, Span<bool>, const Type<ScalarValue>&)>
void register_scalar_rule() {
    scalar_rules[T::typeinfo()] = [](const OpDef& def, Span<ValueRef> inputs,
                                     Span<bool> inputs_mask,
                                     const Type<ScalarValue>& value_type) {
        return (*rule)(def.cast_final_safe<T>(), inputs, inputs_mask, value_type);
    };
}

template <typename TOpDef, size_t nr_inputs>
ValueRefList elemwise_rule(
        const TOpDef& op_def, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    if constexpr (nr_inputs != 0) {
        mgb_assert(inputs.size() == inputs.size(), "inputs size mismatch");
    }
    bool all_scalar = true;
    for (auto&& input_mask : inputs_mask) {
        if (!input_mask) {
            all_scalar = false;
        }
    }
    auto outputs = imperative::apply(op_def, inputs);
    if (all_scalar) {
        outputs[0] = scalar_type.make(outputs[0]);
    }
    return outputs;
}

ValueRefList remove_axis_rule(
        const RemoveAxis& remove_axis, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    mgb_assert(!inputs_mask.item());
    bool is_scalar = inputs.item().shape()->ndim == remove_axis.axis.size();
    if (is_scalar && remove_axis.axis.size() == 1) {
        return {scalar_type.make(inputs.item())};
    }
    auto outputs = imperative::apply(remove_axis, inputs);
    if (is_scalar) {
        outputs[0] = scalar_type.make(outputs[0]);
    }
    return outputs;
}

ValueRefList reduce_rule(
        const Reduce& reduce, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    bool keepdim = reduce.keepdim;
    auto axis = reduce.axis;
    if (inputs.size() == 1) {
        if (axis == INT_MAX || (inputs[0].shape()->ndim == 1 && keepdim == false)) {
            // CompNode device = *inputs[0].device();
            return {scalar_type.make(imperative::apply(reduce, inputs)[0])};
        }
        return imperative::apply(reduce, inputs);
    }
    mgb_assert(inputs.size() == 2);
    bool is_scalar = is_scalar_shape(inputs[1]);
    if (is_scalar) {
        CompNode device = *inputs[0].device();
        return {scalar_type.make(
                imperative::apply(reduce, inputs[0], make_scalar_shape(device))[0])};
    }
    return imperative::apply(reduce, inputs);
}

ValueRefList collective_comm_rule(
        const CollectiveComm& collective_comm, Span<ValueRef> inputs,
        Span<bool> inputs_mask, const Type<ScalarValue>& scalar_type) {
    mgb_assert(inputs.size() == 1);
    static std::unordered_set<CollectiveComm::Mode> modes = {
            CollectiveComm::Mode::ALL_REDUCE_MAX, CollectiveComm::Mode::ALL_REDUCE_MIN,
            CollectiveComm::Mode::ALL_REDUCE_SUM, CollectiveComm::Mode::BROADCAST,
            CollectiveComm::Mode::REDUCE_SUM,
    };
    if (modes.count(collective_comm.mode) == 0) {
        return imperative::apply(collective_comm, inputs);
    }
    if (inputs_mask.item()) {
        return {scalar_type.make(imperative::apply(collective_comm, inputs[0])[0])};
    } else {
        return imperative::apply(collective_comm, inputs);
    }
}

ValueRefList param_pack_split_rule(
        const ParamPackSplit& param_pack_split, Span<ValueRef> inputs,
        Span<bool> inputs_mask, const Type<ScalarValue>& scalar_type) {
    auto outputs = imperative::apply(param_pack_split, inputs);
    size_t nr_outputs = outputs.size();
    mgb_assert(nr_outputs == param_pack_split.shapes.size());
    for (size_t i = 0; i < nr_outputs; ++i) {
        if (param_pack_split.shapes[i].empty()) {
            outputs[i] = scalar_type.make(outputs[i]);
        }
    }
    return outputs;
}

ValueRefList dot_rule(
        const Dot& dot, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    return {scalar_type.make(imperative::apply(dot, inputs)[0])};
}

ValueRefList add_axis_rule(
        const AddAxis& add_axis, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    mgb_assert(inputs.size() == 1);
    if (inputs_mask.item()) {
        mgb_assert(add_axis.axis[0] == 0);
        if (add_axis.axis.size() == 1) {
            return {inputs[0]};
        } else {
            std::vector<int32_t> axis(add_axis.axis.begin() + 1, add_axis.axis.end());
            return imperative::apply(*AddAxis::make(axis, add_axis.scope()), inputs[0]);
        }
    } else {
        return imperative::apply(add_axis, inputs);
    }
}

ValueRefList remote_recv_rule(
        const RemoteRecv& remote_recv, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    if (remote_recv.shape.empty()) {
        std::vector<int32_t> shape = {1};
        auto remote_recv_no_scalar = RemoteRecv::make(
                remote_recv.key, remote_recv.addr, remote_recv.port,
                remote_recv.rank_from, remote_recv.cn, shape, remote_recv.dtype,
                remote_recv.backend);
        remote_recv_no_scalar->set_scope(remote_recv.scope());
        return imperative::apply(ApplyOp(*remote_recv_no_scalar), inputs);
    } else {
        return imperative::apply(remote_recv, inputs);
    }
}

ValueRefList check_no_finite_rule(
        const CheckNonFinite& check_no_finite, Span<ValueRef> inputs,
        Span<bool> inputs_mask, const Type<ScalarValue>& scalar_type) {
    auto outputs = imperative::apply(check_no_finite, inputs);
    mgb_assert(outputs.size() == inputs.size() + 1, "output size mismatch");
    outputs.back() = scalar_type.make(outputs.back());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs_mask[i]) {
            outputs[i] = scalar_type.make(outputs[i]);
        }
    }
    return outputs;
}

ValueRefList subtensor_rule(
        const Subtensor& subtensor, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    mgb_assert(inputs.size() >= 1);
    auto input = inputs[0];
    bool is_scalar;
    mgb_assert(!inputs_mask[0], "subtensor shouldn't have scalar input");
    if (auto shape = input.shape()) {
        size_t ndim = shape->ndim;
        for (auto&& [axis, begin, end, step, idx] : subtensor.items) {
            if (idx) {
                ndim--;
            }
        }
        is_scalar = ndim == 0;
    } else {
        // assume not scalar
        is_scalar = false;
    }
    auto outputs = imperative::apply(subtensor, inputs);
    if (is_scalar) {
        outputs[0] = scalar_type.make(outputs[0]);
    }
    return outputs;
}

ValueRefList get_var_shape_rule(
        const GetVarShape& get_var_shape, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    bool all_scalar = true;
    mgb_assert(inputs.size() >= 1);
    for (auto&& input_mask : inputs_mask) {
        if (!input_mask) {
            all_scalar = false;
        }
    }
    if (all_scalar) {
        auto device = inputs[0].device();
        auto storage = HostStorage::make(*device);
        // storage->ensure_size(1);
        return imperative::apply(
                CreateTensor(
                        CreateTensor::Const, *device, dtype::Int32(), ValueShape{0}),
                storage);
    } else {
        return imperative::apply(get_var_shape, inputs);
    }
}

ValueRefList reshape_rule(
        const Reshape& reshape, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    mgb_assert(inputs.size() == 1 || inputs.size() == 2);
    size_t nr_inp = inputs.size();
    bool is_scalar = (nr_inp == 2 && is_scalar_shape(inputs[1])) ||
                     (nr_inp == 1 && reshape.shape.size() == 0);
    if (is_scalar) {
        return {scalar_type.make(imperative::apply(
                reshape, inputs[0], make_scalar_shape(*inputs[0].device()))[0])};
    } else {
        return imperative::apply(reshape, inputs);
    }
}

ValueRefList broadcast_rule(
        const Broadcast& broadcast, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    mgb_assert(inputs.size() == 1 || inputs.size() == 2);
    size_t nr_inp = inputs.size();
    bool is_scalar = (nr_inp == 2 && is_scalar_shape(inputs[1])) ||
                     (nr_inp == 1 && broadcast.shape.size() == 0);
    if (is_scalar) {
        return {scalar_type.make(imperative::apply(
                broadcast, inputs[0], make_scalar_shape(*inputs[0].device()))[0])};
    } else {
        return imperative::apply(broadcast, inputs);
    }
}

template <typename T>
ValueRefList subgraph_op_rule(
        const T& op, Span<ValueRef> inputs, Span<bool> inputs_mask,
        const Type<ScalarValue>& scalar_type) {
    // TODO: add flag instead of assume
    bool all_scalar = true;
    for (auto&& input_mask : inputs_mask) {
        if (!input_mask) {
            all_scalar = false;
        }
    }
    auto outputs = imperative::apply(op, inputs);
    if (all_scalar) {
        for (auto& output : outputs) {
            output = scalar_type.make(output);
        }
    }
    return outputs;
}

struct ScalarRuleRegistry {
    ScalarRuleRegistry() {
        register_scalar_rule<Elemwise, elemwise_rule<Elemwise, 0>>();
        register_scalar_rule<RemoveAxis, remove_axis_rule>();
        register_scalar_rule<Reduce, reduce_rule>();
        register_scalar_rule<TypeCvt, elemwise_rule<TypeCvt, 1>>();
        register_scalar_rule<CollectiveComm, collective_comm_rule>();
        register_scalar_rule<ParamPackSplit, param_pack_split_rule>();
        register_scalar_rule<Dot, dot_rule>();
        register_scalar_rule<AddAxis, add_axis_rule>();
        register_scalar_rule<RemoteRecv, remote_recv_rule>();
        register_scalar_rule<CheckNonFinite, check_no_finite_rule>();
        register_scalar_rule<Subtensor, subtensor_rule>();
        register_scalar_rule<GetVarShape, get_var_shape_rule>();
        register_scalar_rule<FastpathCopy, elemwise_rule<FastpathCopy, 1>>();
        register_scalar_rule<Reshape, reshape_rule>();
        register_scalar_rule<Broadcast, broadcast_rule>();
        register_scalar_rule<Copy, elemwise_rule<Copy, 1>>();
        register_scalar_rule<InplaceAdd, elemwise_rule<InplaceAdd, 4>>();
        register_scalar_rule<SubgraphOp, subgraph_op_rule<SubgraphOp>>();
        register_scalar_rule<CompiledOp, subgraph_op_rule<CompiledOp>>();
    }
} _;
}  // namespace

ValueRefList ScalarTransformation::apply_get_attr(
        const GetAttr& get_attr, Span<ValueRef> inputs) {
    auto&& input = inputs.item();
    bool is_scalar = input.is(m_value_type);
    if (!is_scalar) {
        return imperative::apply(get_attr, input);
    }
    auto unwrapped_input = input.cast(m_value_type).value();
    if (get_attr.attr() == GetAttr::Shape) {
        if (!m_empty_shape) {
            m_empty_shape = ShapeValue::make();
        }
        return {m_empty_shape};
    } else {
        auto outputs = imperative::apply(get_attr, unwrapped_input);
        auto& output = outputs[0];
        switch (get_attr.attr()) {
            case GetAttr::Value: {
                auto& hv = output.cast<HostValue>();
                mgb_assert(
                        hv.shape() == ValueShape({1}),
                        "underlying value should has shape {1}, got %s",
                        hv.shape().to_string().c_str());
                output = HostValue::make(hv.dtype(), ValueShape(), hv.storage());
                break;
            }
            case GetAttr::Data: {
                auto& dv = output.cast<DeviceValue>();
                mgb_assert(
                        dv.shape() == ValueShape({1}),
                        "underlying value should has shape {1}, got %s",
                        dv.shape().to_string().c_str());
                output = DeviceValue::make(dv.dtype(), ValueShape(), dv.storage());
                break;
            }
            default:
                break;
        }
        return outputs;
    }
}

ValueRefList ScalarTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* get_attr = op.as<GetAttr>()) {
        // fastpath for GetAttr
        return apply_get_attr(*get_attr, inputs);
    } else if (auto* apply_op = op.as<ApplyOp>()) {
        if (apply_op->op().same_type<FastpathCopy>()) {
            return inputs[0];
        }
    }
    size_t nr_inputs = inputs.size();
    ValueRefList unwrapped_inputs(nr_inputs);
    SmallVector<bool> inputs_mask(nr_inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (auto&& scalar_value = inputs[i].as_ref(m_value_type)) {
            unwrapped_inputs[i] = scalar_value->value();
            inputs_mask[i] = true;
        } else {
            unwrapped_inputs[i] = inputs[i];
            inputs_mask[i] = false;
        }
    }
    auto fallback = [&] { return imperative::apply(op, unwrapped_inputs); };
    if (auto apply_op = op.as<ApplyOp>()) {
        auto iter = scalar_rules.find(apply_op->op().dyn_typeinfo());
        if (iter != scalar_rules.end()) {
            return iter->second(
                    apply_op->op(), unwrapped_inputs, inputs_mask, m_value_type);
        } else {
            // TODO: repeat op
            return fallback();
        }
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        if (create_tensor->shape().is_scalar()) {
            ValueShape scalar_shape = {1};
            CreateTensor scalar_op(
                    create_tensor->kind(), create_tensor->device(),
                    create_tensor->dtype(), scalar_shape);
            return {m_value_type.make(imperative::apply(scalar_op, inputs)[0])};
        } else {
            return imperative::apply(op, inputs);
        }
    } else if (op.as<IsScalar>()) {
        mgb_assert(nr_inputs == 1);
        return {BoolValue::make(inputs_mask[0])};
    } else if (op.is<Operator::IdentityLike>()) {
        mgb_assert(nr_inputs == 1);
        bool is_scalar = inputs_mask[0];
        auto outputs = fallback();
        if (is_scalar) {
            outputs[0] = m_value_type.make(outputs[0]);
        }
        return outputs;
    } else {
        return fallback();
    }
};

}  // namespace imperative
}  // namespace mgb
