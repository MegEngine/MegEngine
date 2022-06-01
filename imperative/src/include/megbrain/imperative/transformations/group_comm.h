/**
 * \file imperative/src/include/megbrain/imperative/scalar.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb::imperative {

class PlaceholderValue final : public ObjectValue<PlaceholderValue> {
public:
    std::string to_string() const override { return ssprintf("PlaceholderValue"); }
    void clear() override {}
};

class GroupCommTransformation final : public Transformation {
private:
    SmallVector<ValueRef> send_inputs;
    std::vector<PlaceholderValue::weak_ref_t> recv_tensors;
    SmallVector<std::shared_ptr<OpDef>> record_ops;
    ObjectType<PlaceholderValue> m_value_type{"PlaceholderValue"};

public:
    GroupCommTransformation() = default;
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;
    ValueRefList execute_batch_op();
    ValueRef unwrap(ValueRef value) override { return value; }
    std::string name() const override { return "GroupCommTransformation"; }
    void on_unregister() noexcept override;
    ~GroupCommTransformation();
};

}  // namespace mgb::imperative
