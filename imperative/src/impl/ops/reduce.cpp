/**
 * \file imperative/src/impl/ops/reduce.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/opr/basic_arith.h"

#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {
namespace {
namespace reduce {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& reduce = static_cast<const Reduce&>(def);
    OperatorNodeConfig config{reduce.make_name()};
    if (inputs.size() > 1) {
        return opr::Reduce::make(inputs[0], reduce.param(), inputs[1], config);
    } else {
        return opr::Reduce::make(
                inputs[0], reduce.param(), (cg::VarNode*)nullptr, config);
    }
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Reduce>();
    return Reduce::make(node->param());
}

// TODO: using this for apply_on_physical_tensor
bool memory_forward_success(const OpDef& def, SmallVector<TensorPtr> inputs) {
    auto&& reduce = static_cast<const Reduce&>(def);
    if (reduce.mode != Reduce::Mode::SUM_SQR && inputs.size() == 2) {
        auto shape_tensor = inputs[1]->get_value();
        TensorShape shape;
        cg::copy_tensor_value_to_shape(shape, shape_tensor.proxy_to_default_cpu());
        if (shape.eq_shape(inputs[0]->shape())) {
            return true;
        }
    }
    return false;
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    if (memory_forward_success(def, inputs)) {
        return {Tensor::make(
                inputs[0]->blob(), inputs[0]->offset(), inputs[0]->layout())};
    }
    return proxy_graph_detail::apply_on_physical_tensor(
            def, inputs, output_descs, validated);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto [output_descs, validated] =
            proxy_graph_detail::infer_output_attrs_fallible(def, inputs);
    if (inputs.size() == 2 && !output_descs[0].layout.ndim) {
        if (!inputs[1].value.empty()) {
            cg::copy_tensor_value_to_shape(output_descs[0].layout, inputs[1].value);
            output_descs[0].layout.init_contiguous_stride();
        }
    }
    return {output_descs, validated};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(Reduce, Reduce, opr::Reduce)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace reduce
}  // namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
