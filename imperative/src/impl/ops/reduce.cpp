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
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    if (memory_forward_success(def, inputs)) {
        return {Tensor::make(inputs[0]->blob(), 0, inputs[0]->layout())};
    }
    return proxy_graph_detail::apply_on_physical_tensor(def, inputs);
}

OP_TRAIT_REG(Reduce, Reduce, opr::Reduce)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace reduce
}  // namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
