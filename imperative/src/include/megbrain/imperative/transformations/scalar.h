#pragma once

#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb::imperative {

class ScalarValue final : public ObjectValue<ScalarValue> {
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
    ObjectType<ScalarValue> m_value_type{"ScalarValue"};

public:
    ValueRefList apply_get_attr(const GetAttr& get_attr, Span<ValueRef> inputs);
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is(m_value_type));
        return value;
    }

    std::string name() const override { return "ScalarTransformation"; }

    const Type<ScalarValue>& value_type() const { return m_value_type; }
};

}  // namespace mgb::imperative
