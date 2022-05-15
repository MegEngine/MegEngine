#pragma once

#include <future>
#include <variant>

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/opr/io.h"

namespace mgb::imperative {

class SymbolValue final : public ObjectValue<SymbolValue> {
private:
    VarNode* m_node = nullptr;

public:
    SymbolValue(VarNode* node) : m_node(node) {}

    VarNode* node() const { return m_node; }

    std::string to_string() const override { return ssprintf("VarNode{%p}", m_node); }

    void clear() override { m_node = nullptr; }
};

/**
 * \brief this transformation is used to handle VarNode.
 *
 * Unlike other transformations, this transformation is not used in Tensor evaluation.
 * when user calls py_apply(SymbolVar), we'll switch current transformation context to a
 * special symbol context. The advantage is that we can handle scalar by
 * ScalarTransformation.
 */
class SymbolTransformation final : public Transformation {
private:
    ComputingGraph* m_graph = nullptr;
    ObjectType<SymbolValue> m_value_type{"SymbolValue"};

public:
    SymbolTransformation() {}
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override {
        ComputingGraph* cg = nullptr;
        if (auto* node_value = op.as<CreateNode>()) {
            return {m_value_type.make(node_value->node())};
        }
        for (auto&& input : inputs) {
            if (auto* val = input.as(m_value_type)) {
                auto* node = val->node();
                ComputingGraph* cur_cg = node->owner_graph();
                if (cg == nullptr) {
                    cg = cur_cg;
                } else {
                    mgb_assert(cg == cur_cg, "input varnode gragh should be the same");
                }
            }
        }
        if (!cg) {
            return imperative::apply(op, inputs);
        }

        if (auto* apply_op = op.as<ApplyOp>()) {
            SmallVector<VarNode*> input_nodes;
            for (auto&& input : inputs) {
                if (!input.is(m_value_type)) {
                    auto* node = opr::ImmutableTensor::make(
                                         *cg, input.numpy()->as_nd(true), {})
                                         .node();
                    input_nodes.push_back(node);
                } else {
                    input_nodes.push_back(input.cast(m_value_type).node());
                }
            }
            auto output_nodes = OpDef::apply_on_var_node(apply_op->op(), input_nodes);
            ValueRefList outputs(output_nodes.size());
            for (size_t i = 0; i < output_nodes.size(); ++i) {
                outputs[i] = m_value_type.make(output_nodes[i]);
            }
            return outputs;
        } else if (auto* get_attr = op.as<GetAttr>()) {
            auto* node = inputs.item().cast(m_value_type).node();
            auto* m_graph = node->owner_graph();
            switch (get_attr->attr()) {
                case GetAttr::DType:
                    return {DTypeValue::make(node->dtype())};
                case GetAttr::Device:
                    return {CompNodeValue::make(node->comp_node())};
                case GetAttr::Shape: {
                    if (!cg::is_static_var_shape(node)) {
                        mgb_log_debug(
                                "shape inference invalid for %s", node->name().c_str());
                        return {ValueRef()};
                    }
                    auto shape = m_graph->static_infer_manager().infer_shape(node);
                    return {ShapeValue::make(ValueShape::from(shape))};
                }
                case GetAttr::Value: {
                    if (!cg::is_static_var_value(node)) {
                        mgb_log_debug(
                                "value inference invalid for %s", node->name().c_str());
                        return {ValueRef()};
                    }
                    auto inferred_value =
                            m_graph->static_infer_manager().infer_value(node);
                    HostTensorND host_value(node->comp_node(), node->dtype());
                    host_value.copy_from(inferred_value);
                    return {HostValue::make(host_value)};
                }
                case GetAttr::Data: {
                    if (!cg::is_static_var_value(node)) {
                        mgb_log_debug(
                                "value inference invalid for %s", node->name().c_str());
                        return {ValueRef()};
                    }
                    auto inferred_value =
                            m_graph->static_infer_manager().infer_value(node);
                    DeviceTensorND dev_value(node->comp_node(), node->dtype());
                    dev_value.copy_from(inferred_value);
                    return {DeviceValue::make(dev_value)};
                }
                default:
                    mgb_throw(
                            MegBrainError, "Symbol: malformed GetAttr: %s",
                            op.to_string().c_str());
            }
        } else if (auto* get_attr = op.as<GetVarVal>()) {
            cg::VarNode* node = inputs.item().cast(m_value_type).node();
            NodeStorage inp_var = NodeStorage(node);
            return {NodeValue::make(inp_var)};
        } else {
            return op.fallback(inputs);
        }
    }

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is(m_value_type), "SymbolValue doesn't support unwrap");
        return value;
    }

    std::string name() const override { return "SymbolTransformation"; }

    const Type<SymbolValue>& value_type() const { return m_value_type; }
};

}  // namespace mgb::imperative
