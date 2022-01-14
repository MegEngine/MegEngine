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

#include "megbrain/imperative/transformations/eval.h"
#include "megbrain/imperative/transformations/grad.h"

namespace mgb {
namespace imperative {

std::vector<ValueRef> InterpreterTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* op_val = op.as<ApplyOp>()) {
        if (op_val->op().same_type<FastpathCopy>()) {
            return {inputs[0]};
        }
        SmallVector<Handle> input_handles;
        SmallVector<Handle> output_handles;
        CleanupGuard _{[&] {
            for (auto handle : output_handles) {
                if (handle) {
                    m_channel->del(handle);
                }
            }
        }};
        for (auto input : inputs) {
            input_handles.push_back(*input.cast<InterpreterValue>().handle());
        }
        output_handles =
                m_channel->apply_op(op_val->op().shared_from_this(), input_handles);
        std::vector<ValueRef> outputs;
        for (auto& handle : output_handles) {
            outputs.push_back(InterpreterValue::make(share_handle(handle)));
            handle = nullptr;
        }
        return outputs;
    } else if (auto* get_attr = op.as<GetAttr>()) {
        Handle handle = *inputs[0].cast<InterpreterValue>().handle();
        ValueRef output;
        switch (get_attr->attr()) {
            case GetAttr::DType:
                output = DTypeValue::make(m_channel->get_dtype(handle));
                break;
            case GetAttr::Shape:
                output = ShapeValue::make(
                        ValueShape::from(m_channel->get_shape(handle)));
                break;
            case GetAttr::Device:
                output = CompNodeValue::make(m_channel->get_device(handle));
                break;
            case GetAttr::Value:
                output = HostValue::make(m_channel->get_value(handle));
                break;
            case GetAttr::Data:
                output = DeviceValue::make(m_channel->get_dev_tensor(handle));
                break;
            default:
                mgb_throw(
                        MegBrainError, "Interpreter: malformed GetAttr: %s",
                        op.to_string().c_str());
        }
        return {output};
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        auto args = create_tensor->parse(inputs);
        if (!args.device) {
            // implies H2D
            mgb_assert(args.host, "neither host and device value is valid");
            return {InterpreterValue::make(share_handle(
                    m_channel->put(*args.host, args.kind == CreateTensor::Unique)))};
        } else {
            return {InterpreterValue::make(share_handle(m_channel->put(
                    *args.device, args.host ? *args.host : HostTensorND())))};
        }
    } else if (auto* dtr_command = op.as<DTRCommand>()) {
        auto handle = *inputs[0].cast<InterpreterValue>().handle();
        switch (dtr_command->kind()) {
            case DTRCommand::Drop:
                m_channel->drop(handle);
                break;
            default:
                mgb_throw(AssertionError, "unknown DTRCommand %d", dtr_command->kind());
        }
        return {};
    } else if (auto* rename_value = op.as<RenameValue>()) {
        auto& input = inputs[0].cast<InterpreterValue>();
        return {InterpreterValue::make(input.handle(), rename_value->name())};
    } else if (op.is<GetName>()) {
        auto name = inputs[0].cast<InterpreterValue>().name();
        if (!name.empty()) {
            return {StringValue::make(name)};
        } else {
            return {ValueRef()};
        }
    } else {
        return imperative::apply(op, inputs);
    }
}

}  // namespace imperative
}  // namespace mgb
