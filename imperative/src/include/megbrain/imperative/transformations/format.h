#pragma once

#include "megbrain/imperative/basic_values.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/data_format.h"

namespace mgb::imperative {

class FormattedTensorValue final : public ObjectValue<FormattedTensorValue> {
private:
    ValueRef m_value;
    Format m_format;

public:
    FormattedTensorValue(ValueRef value, Format format)
            : m_value(value), m_format(format) {}

    std::string to_string() const override {
        return ssprintf(
                "FormattedTensorValue{value=%s, format=%s}",
                m_value.to_string().c_str(), m_format.to_string().c_str());
    }

    ValueRef value() const { return m_value; }

    const Format& format() const { return m_format; }

    void set_format(Format format) { m_format = format; }

    void clear() override {
        m_value = {};
        m_format = {};
    }

    void on_watch() override { m_value.watch(); }

    void on_unwatch() override { m_value.unwatch(); }
};

class FormatTransformation final : public Transformation {
private:
    // enable auto_convert by default to be easier to use.
    bool m_auto_convert = true;
    ObjectType<FormattedTensorValue> m_value_type{"FormattedTensorValue"};

public:
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        // mgb_assert(!value.is(m_value_type));
        if (auto format_val = value.as_ref(m_value_type)) {
            return format_val->value();
        }
        return value;
    }

    std::string name() const override {
        return ssprintf("FormatTransformation{auto_convert=%d}", m_auto_convert);
    }
    void set_auto_convert(bool enabled) { m_auto_convert = enabled; }
    bool get_auto_convert() const { return m_auto_convert; }

    const Type<FormattedTensorValue>& value_type() const { return m_value_type; }

    inline ValueRef unwrap_input(const ValueRef& input) const;
    inline ValueRefList unwrap_inputs(const Span<ValueRef>& inputs) const;
    inline ValueRef wrap_output(
            const ValueRef& output, Format format = Format::Type::DEFAULT) const;
    inline ValueRefList wrap_outputs(
            const ValueRefList& outputs, Format format = Format::Type::DEFAULT) const;

    TypedValueRef<FormattedTensorValue> as(
            const FormattedTensorValue&, const Format::Type& target) const;
    TypedValueRef<FormattedTensorValue> to(
            const FormattedTensorValue&, const Format::Type& target,
            const std::string& scope = "") const;
};

}  // namespace mgb::imperative
