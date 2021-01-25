/**
 * \file imperative/src/impl/ops/backward_graph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <sstream>
#include <range/v3/all.hpp>

#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
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

std::tuple<SmallVector<LogicalTensorDesc>, bool> BackwardGraph::InternalGraph::infer_attrs(
        const SmallVector<LogicalTensorDesc>& inputs) const {
    using TensorAttr = LogicalTensorDesc;
    ThinHashMap<size_t, TensorAttr> node2attr;
    auto&& input_nodes = this->inputs;
    auto&& output_nodes = this->outputs;
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
    bool validated = true;
    for (size_t i = 0; i < exprs.size(); ++ i) {
        auto&& [expr_op, expr_inps, expr_oups] = exprs[i];
        SmallVector<TensorAttr> expr_input_descs;
        for (auto &&inp : expr_inps) {
            expr_input_descs.push_back(node2attr.at(inp));
        }

        auto [expr_output_descs, expr_validated] = OpDef::infer_output_attrs_fallible(
            *expr_op, expr_input_descs);
        validated = validated && expr_validated;

        mgb_assert(expr_output_descs.size() == expr_oups.size());
        for (size_t i = 0; i < expr_output_descs.size(); ++ i) {
            node2attr[expr_oups[i]] = expr_output_descs[i];
        }
    }

    SmallVector<TensorAttr> ret;
    for (auto &&i : output_nodes) {
        ret.push_back(node2attr.at(i));
    }
    return {ret, validated};
}

std::string BackwardGraph::InternalGraph::repr() {
    std::ostringstream buf;
    buf << "(";
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0) buf << ", ";
        buf << "%" << inputs[i];
    }
    buf << ") => {\n";
    auto fmt_const = [](size_t i, TensorPtr& t) {
        if (t->shape().ndim == 1 && t->shape()[0] == 1) {
            auto&& v = t->get_value();
            if (v.dtype() == dtype::Float32{}) {
                return std::to_string(*v.ptr<dt_float32>());
            } else if (v.dtype() == dtype::Int32{}) {
                return std::to_string(*v.ptr<int32_t>());
            }
        }
        return std::string("%c") + std::to_string(i);
    };
    std::unordered_map<size_t, std::string> const_reps;
    for (auto&& [i, t] : constants) {
        const_reps.emplace(i, fmt_const(i, t));
    }
    for (auto& [op, ins, outs] : exprs) {
        buf << "  ";
        if (outs.size()) {
            for (size_t i = 0; i < outs.size(); ++i) {
                if (i > 0) buf << ", ";
                buf << "%" << outs[i];
            }
            buf << " = ";
        }
        if (auto* p = op->try_cast_final<OprAttr>()) {
            buf << p->type;
        } else {
            buf << op->dyn_typeinfo()->name;
        }
        for (size_t i : ins) {
            buf << " ";
            auto&& it = const_reps.find(i);
            if (it != const_reps.end()) {
                buf << it->second;
            } else {
                buf << "%" << i;
            }
        }
        buf << "\n";
    }
    buf << "  ";
    if (outputs.size()) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i > 0) buf << ", ";
            buf << "%" << outputs[i];
        }
    } else {
        buf << "()";
    }
    buf << "\n}\n";
    return buf.str();
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BackwardGraph);

namespace {
SmallVector<TensorPtr> backward_impl(
    const OpDef& backward_graph,
    const SmallVector<TensorPtr>& tensors) {
    return backward_graph.cast_final_safe<BackwardGraph>()
            .graph().apply(tensors);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_tensor_attrs(
    const OpDef& backward_graph,
    const SmallVector<LogicalTensorDesc> inputs) {
    return backward_graph.cast_final_safe<BackwardGraph>()
        .graph().infer_attrs(inputs);
}

std::vector<std::pair<const char*, std::string>> props(
    const OpDef& backward_graph) {
    return {};
}

OP_TRAIT_REG(BackwardGraph, BackwardGraph)
    .apply_on_physical_tensor(backward_impl)
    .infer_output_attrs_fallible(infer_tensor_attrs)
    .props(props)
    .fallback();
} // anonymous namespace

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
