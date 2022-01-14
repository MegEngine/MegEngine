/**
 * \file imperative/src/impl/dispatch.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/dispatch.h"

#include "megbrain/imperative/utils/debug.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/imperative/utils/map.h"

namespace mgb {
namespace imperative {

std::vector<ValueRef> apply(const Operator& op, Span<ValueRef> inputs) {
    static bool log_dispatch = MGB_GETENV("MGE_LOG_OP_DISPATCH");
    bool enable_watch = ValueRef::any_watching();
    auto& context = Transformation::get_context();
    size_t& depth = context.next_transformation;
    static const char tabs_storage[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    const char* tabs = tabs_storage + sizeof(tabs_storage) / sizeof(char) - depth - 1;
    bool log_current_dispatch = log_dispatch;
    if (enable_watch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& input = inputs[i];
            if (input.watching()) {
                log_current_dispatch = true;
                mgb_log_debug("%sinput[%zu] is %s", tabs, i, input.to_string().c_str());
                debug::notify_event("apply");
            }
        }
    }
    // entrance
    std::vector<ValueRef> outputs;
    if (depth >= context.transformations.size()) {
        // fallback
        if (log_current_dispatch) {
            mgb_log_debug(
                    "%sfallback apply %s in %s", tabs, op.to_string().c_str(),
                    imperative::to_string(inputs).c_str());
        }
        outputs = op.fallback(inputs);
    } else {
        // dispatch to stack top
        auto& transformation = *context.transformations[depth];
        ++depth;
        context.frames.push_back({op, inputs});
        CleanupGuard _{[&] {
            context.frames.pop_back();
            --depth;
        }};
        if (log_current_dispatch) {
            mgb_log_debug(
                    "%s%s apply %s in %s", tabs, transformation.name().c_str(),
                    op.to_string().c_str(), imperative::to_string(inputs).c_str());
        }
        outputs = transformation.apply_transformation(op, inputs);
    }
    if (log_current_dispatch) {
        mgb_log_debug("%sreturn %s", tabs, imperative::to_string(outputs).c_str());
    }
    return outputs;
}

std::vector<ValueRef> apply(const OpDef& def, Span<ValueRef> inputs) {
    return imperative::apply(ApplyOp{def}, inputs);
}

std::vector<ValueRef> apply(Subgraph graph, Span<ValueRef> inputs) {
    SmallVector<ValueRef> inputs_storage;
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs_storage.push_back(inputs[i]);
    }
    auto apply_functor = [](std::shared_ptr<OpDef> op, SmallVector<ValueRef> inputs,
                            size_t) {
        auto outputs = imperative::apply(ApplyOp(*op), inputs);
        return SmallVector<ValueRef>(outputs.begin(), outputs.end());
    };
    auto make_const = [](TensorPtr constant) -> ValueRef {
        auto host_value = constant->get_value();
        auto device_value = constant->dev_tensor();
        mgb_assert(
                host_value.layout().is_contiguous() &&
                device_value.layout().is_contiguous());
        ValueShape shape;
        // FIXME: assume Tensor with shape {1} is scalar
        if (!constant->shape().is_scalar()) {
            shape = ValueShape::from(constant->shape());
        }
        return imperative::apply(
                CreateTensor(
                        CreateTensor::Const, constant->comp_node(), constant->dtype(),
                        shape),
                HostStorage::make(host_value.storage()),
                DeviceStorage::make(device_value.storage()))[0];
    };
    auto outputs = graph.apply(inputs_storage, apply_functor, make_const);
    return {outputs.begin(), outputs.end()};
}

}  // namespace imperative
}  // namespace mgb
