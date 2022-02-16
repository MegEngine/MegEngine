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
#include "megbrain/imperative/utils/stats.h"

namespace mgb {
namespace imperative {

ValueRefList apply(const Operator& op, Span<ValueRef> inputs) {
    auto& context = Transformation::get_context();
    size_t& depth = context.next_transformation;
    // TODO: add fallback transformation
    bool fallback = depth >= context.transformations.size();
    if (mgb_unlikely(fallback)) {
        return op.fallback(inputs);
    } else {
        auto& transformation = *context.transformations[depth++];
        CleanupGuard _{[&] { --depth; }};
        return transformation.apply_transformation(op, inputs);
    }
}

ValueRefList apply(const OpDef& def, Span<ValueRef> inputs) {
    return imperative::apply(ApplyOp{def}, inputs);
}

ValueRefList apply(const Subgraph& graph, Span<ValueRef> inputs) {
    auto apply_functor = [](std::shared_ptr<OpDef> op, Span<ValueRef> inputs, size_t) {
        auto outputs = imperative::apply(*op, inputs);
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
    auto outputs = graph.apply(inputs, apply_functor, make_const);
    return ValueRefList{outputs.begin(), outputs.end()};
}

}  // namespace imperative
}  // namespace mgb
