/**
 * \file imperative/src/include/megbrain/imperative/symbol.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <future>
#include <variant>

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/opr/io.h"

namespace mgb::imperative {

class SymbolValue final : public ValueImpl<SymbolValue> {
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

public:
    SymbolTransformation(ComputingGraph* graph) : m_graph(graph) {}
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override {
        if (auto* apply_op = op.as<ApplyOp>()) {
            SmallVector<VarNode*> input_nodes;
            for (auto&& input : inputs) {
                input_nodes.push_back(input.cast<SymbolValue>().node());
            }
            auto output_nodes = OpDef::apply_on_var_node(apply_op->op(), input_nodes);
            ValueRefList outputs(output_nodes.size());
            for (size_t i = 0; i < output_nodes.size(); ++i) {
                outputs[i] = SymbolValue::make(output_nodes[i]);
            }
            return outputs;
        } else if (auto* create_tensor = op.as<CreateTensor>()) {
            auto&& args = create_tensor->parse(inputs);
            mgb_assert(
                    args.kind == CreateTensor::Const,
                    "only const value is allowed here");
            auto* node = opr::ImmutableTensor::make(*m_graph, *args.host, {}).node();
            return {SymbolValue::make(node)};
        } else if (auto* get_attr = op.as<GetAttr>()) {
            auto* node = inputs.as_array<1>()[0].cast<SymbolValue>().node();
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
        } else {
            return op.fallback(inputs);
        }
    }

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is<SymbolValue>(), "SymbolValue doesn't support unwrap");
        return value;
    }

    std::string name() const override { return "SymbolTransformation"; }
};

}  // namespace mgb::imperative
