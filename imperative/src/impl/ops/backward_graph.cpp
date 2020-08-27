/**
 * \file src/core/impl/imperative/physical_tensor.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/imperative/ops/backward_graph.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {

SmallVector<TensorPtr>
BackwardGraph::InternalGraph::apply(
        const SmallVector<TensorPtr>& inputs) const {
    ThinHashMap<size_t, TensorPtr> node2tensor;
    auto&& input_nodes = this->inputs;
    mgb_assert(inputs.size() == input_nodes.size());
    for (size_t i = 0; i < inputs.size(); ++ i) {
        node2tensor[input_nodes[i]] = inputs[i];
    }
    for (auto &&i : constants) {
        node2tensor[i.first] = i.second;
    }
    for (size_t i = 0; i < exprs.size(); ++ i) {
        auto&& expr = exprs[i];
        SmallVector<TensorPtr> inputs;
        for (auto &&in : std::get<1>(expr)) {
            inputs.push_back(node2tensor.at(in));
        }
        auto outputs = OpDef::apply_on_physical_tensor(
                *std::get<0>(expr), inputs);
        auto output_nodes = std::get<2>(expr);
        mgb_assert(outputs.size() == output_nodes.size());
        for (size_t i = 0; i < outputs.size(); ++ i) {
            node2tensor[output_nodes[i]] = outputs[i];
        }
    }
    SmallVector<TensorPtr> ret;
    for (auto &&i : outputs) {
        ret.push_back(node2tensor.at(i));
    }
    return ret;
}

SmallVector<LogicalTensorDesc>
BackwardGraph::InternalGraph::infer_attrs(
        const SmallVector<LogicalTensorDesc>& inputs) const {
    using TensorAttr = LogicalTensorDesc;
    ThinHashMap<size_t, TensorAttr> node2attr;
    auto&& input_nodes = this->inputs;
    mgb_assert(inputs.size() == input_nodes.size());
    for (size_t i = 0; i < inputs.size(); ++ i) {
        node2attr[input_nodes[i]] = inputs[i];
    }
    for (auto &&i : constants) {
        auto* value = i.second->try_get_value();
        mgb_assert(value);
        node2attr[i.first] = TensorAttr{
            i.second->layout(), i.second->comp_node(),
            value->proxy_to_default_cpu()};
    }
    for (size_t i = 0; i < exprs.size(); ++ i) {
        auto&& expr = exprs[i];
        SmallVector<TensorAttr> inputs;
        for (auto &&in : std::get<1>(expr)) {
            inputs.push_back(node2attr.at(in));
        }
        auto outputs = OpDef::infer_output_attrs_fallible(
                *std::get<0>(expr), inputs);
        auto output_nodes = std::get<2>(expr);
        mgb_assert(outputs.size() == output_nodes.size());
        for (size_t i = 0; i < outputs.size(); ++ i) {
            node2attr[output_nodes[i]] = outputs[i];
        }
    }
    SmallVector<TensorAttr> ret;
    for (auto &&i : outputs) {
        ret.push_back(node2attr.at(i));
    }
    return ret;
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BackwardGraph);

namespace {
SmallVector<TensorPtr> backward_impl(
    const OpDef& backward_graph,
    const SmallVector<TensorPtr>& tensors) {
    return backward_graph.cast_final_safe<BackwardGraph>()
            .graph().apply(tensors);
}

SmallVector<LogicalTensorDesc> infer_tensor_attrs(
    const OpDef& backward_graph,
    const SmallVector<LogicalTensorDesc> inputs) {
    return backward_graph.cast_final_safe<BackwardGraph>()
            .graph().infer_attrs(inputs);
}

OP_TRAIT_REG(BackwardGraph, BackwardGraph)
    .apply_on_physical_tensor(backward_impl)
    .infer_output_attrs_fallible(infer_tensor_attrs)
    .fallback();
} // anonymous namespace

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
