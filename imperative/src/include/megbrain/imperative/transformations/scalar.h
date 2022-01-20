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

class ScalarValue final : public ValueImpl<ScalarValue> {
private:
    ValueRef m_value;

public:
    ScalarValue(ValueRef value) : m_value(value) {}

    std::string to_string() const override {
        return ssprintf("ScalarValue{value=%s}", m_value.to_string().c_str());
    }

    ValueRef value() const { return m_value; }

    void clear() override { m_value = {}; }

    void on_watch() override { m_value.watch(); }

    void on_unwatch() override { m_value.unwatch(); }
};

/**
 * \brief simulates scalar because megbrain graph system don't support scalar
 *
 * Assume that we has 'a = ScalarValue(b)', thus 'a.shape == []', 'b.shape == [1]'.
 * This transformation simulates scalars with a flag. If a value is ScalarValue, it is
 * scalar, vice versa. So there is not scalar down this layer.
 */
class ScalarTransformation final : public Transformation {
private:
    ShapeValue::ref_t m_empty_shape;  // []
public:
    ValueRefList apply_get_attr(const GetAttr& get_attr, Span<ValueRef> inputs);
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is<ScalarValue>());
        return value;
    }

    std::string name() const override { return "ScalarTransformation"; }
};

}  // namespace mgb::imperative
