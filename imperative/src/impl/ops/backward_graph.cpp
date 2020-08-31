/**
 * \file imperative/src/impl/ops/backward_graph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/backward_graph.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {

SmallVector<TensorPtr>
BackwardGraph::InternalGraph::apply(
        const SmallVector<TensorPtr>& inputs) const {
    return interpret<TensorPtr>(
        &OpDef::apply_on_physical_tensor,
        [](const TensorPtr& x) {return x;},
        inputs);
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
