#pragma once

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/value.h"

namespace mgb::imperative {

class DimExpansionTransformation final : public Transformation {
private:
public:
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;
    ValueRef unwrap(ValueRef value) override;
    std::string name() const override;
    void on_register() override;
    void on_unregister() noexcept override;
};

}  // namespace mgb::imperative
