#pragma once

#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/operator.h"
#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/value.h"

namespace mgb::imperative {

struct TangentInfo {
    ValueRef value;
    ValueRef tangent;
};

class TangentTransformation final : public Transformation {
public:
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override { mgb_assert(false); }

    std::string name() const override { return "Tangent"; }
};

}  // namespace mgb::imperative
