#include "megbrain/imperative/transformations/dim_expansion.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb::imperative {

namespace {
using DimExpansionRule = std::function<ValueRefList(const OpDef&, Span<ValueRef>)>;
static std::unordered_map<Typeinfo*, DimExpansionRule> dim_expansion_rules;

template <typename T>
void register_dim_expansion_rules(const DimExpansionRule& rule) {
    dim_expansion_rules[T::typeinfo()] = [rule](const OpDef& def,
                                                Span<ValueRef> inputs) {
        return rule(def.cast_final_safe<T>(), inputs);
    };
}

ValueRefList conv1d_rule(const OpDef& op, Span<ValueRef> inputs) {
    bool need_expand = inputs.at(0).shape()->ndim == 3;
    if (!need_expand)
        return imperative::apply(op, inputs);

    ValueRefList converted(inputs.size());
    std::vector<int32_t> axis = {(int32_t)3};
    for (size_t i = 0; i < inputs.size(); ++i) {
        converted[i] = imperative::apply(ApplyOp(*AddAxis::make(axis)), inputs[i])[0];
    }

    auto outputs = imperative::apply(op, converted);
    outputs[0] = imperative::apply(ApplyOp(*RemoveAxis::make(axis)), outputs[0])[0];
    return outputs;
}

ValueRefList bn1d_rule(const OpDef& op, Span<ValueRef> inputs) {
    size_t ndim = inputs.at(0).shape()->ndim;
    bool need_expand = (ndim == 2 || ndim == 3);
    if (!need_expand)
        return imperative::apply(op, inputs);

    ValueRefList converted(inputs.size());
    std::vector<int32_t> axis = {(int32_t)3};
    if (ndim == 2) {
        axis.insert(axis.begin(), (int32_t)2);
    }
    converted[0] = imperative::apply(ApplyOp(*AddAxis::make(axis)), inputs[0])[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        converted[i] = inputs[i];
    }

    std::reverse(std::begin(axis), std::end(axis));
    auto outputs = imperative::apply(op, converted);
    size_t idx = outputs.size() - 1;
    outputs[idx] = imperative::apply(ApplyOp(*RemoveAxis::make(axis)), outputs[idx])[0];
    return outputs;
}

struct DimExpansionRuleRegistry {
    DimExpansionRuleRegistry() {
        register_dim_expansion_rules<Convolution>(conv1d_rule);
        register_dim_expansion_rules<BatchNorm>(bn1d_rule);
    }
} register_helper;

}  // namespace

ValueRefList DimExpansionTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto apply_op = op.as<ApplyOp>()) {
        auto iter = dim_expansion_rules.find(apply_op->op().dyn_typeinfo());
        if (iter != dim_expansion_rules.end()) {
            return iter->second(apply_op->op(), inputs);
        } else {
            return imperative::apply(op, inputs);
        }
    }
    return imperative::apply(op, inputs);
}

ValueRef DimExpansionTransformation::unwrap(ValueRef value) {
    return value;
}

std::string DimExpansionTransformation::name() const {
    return "DimExpansionTransformation";
}

void DimExpansionTransformation::on_register() {
    // printf("DimExpansionTransformation has been registered\n");
}

void DimExpansionTransformation::on_unregister() noexcept {
    // printf("DimExpansionTransformation has been unregistered\n");
}

}  // namespace mgb::imperative