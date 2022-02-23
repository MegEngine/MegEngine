#pragma once

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/value.h"

namespace mgb::imperative {

class DTypePromoteTransformation final : public Transformation {
private:
public:
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;
    ValueRef unwrap(ValueRef value) override;
    std::string name() const override;
    void on_register() override;
    void on_unregister() noexcept override;
};

struct DTypePromoteCfg {
    static bool convert_input_enabled;
    static bool amp_dtype_autocast_enabled;
    static DType amp_high_prec_dtype;
    static DType amp_low_prec_dtype;
};

}  // namespace mgb::imperative
