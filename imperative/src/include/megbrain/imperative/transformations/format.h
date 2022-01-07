#pragma once

#include "megbrain/imperative/basic_values.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/data_format.h"

namespace mgb::imperative {

class FormattedTensorValue final : public ValueImpl<FormattedTensorValue> {
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

    TypedValueRef<FormattedTensorValue> as(const Format::Type& target) const;
    TypedValueRef<FormattedTensorValue> to(
            const Format::Type& target, const std::string& scope = "") const;

    void clear() override {
        m_value = {};
        m_format = {};
    }

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
class FormatTransformation final : public Transformation {
private:
    bool m_auto_convert = false;

public:
    std::vector<ValueRef> apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is<FormattedTensorValue>());
        return value;
    }

    std::string name() const override {
        return ssprintf("FormatTransformation{auto_convert=%d}", m_auto_convert);
    }
    void set_auto_convert(bool enabled) { m_auto_convert = enabled; }
    bool get_auto_convert() const { return m_auto_convert; }
};

}  // namespace mgb::imperative
