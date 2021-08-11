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
#include "megbrain/opr/basic_arith.h"
#include "megbrain/imperative/proxy_graph_detail.h"

#include "../op_trait.h"
#include "../dnn_op_helper.h"

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
        return opr::Reduce::make(inputs[0], reduce.param(),
                                 (cg::VarNode*)nullptr, config);
    }
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Reduce>();
    return Reduce::make(node->param());
}

bool memory_forward_success(
        const OpDef& def,
        SmallVector<TensorPtr> inputs) {
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

std::tuple<SmallVector<MemoryDesc>, SmallVector<MemoryDesc>> infer_output_mem_desc(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs_tensors,
        const SmallVector<MemoryDesc>& inputs_mems) {
    if (memory_forward_success(def, inputs_tensors)) {
        auto& src_desc = inputs_mems[0];
        return {{{src_desc.layout, 0, src_desc.cn, StorageIdentifier::make(&src_desc)}}, {}};
    }
    return proxy_graph_detail::infer_output_mem_desc(def, inputs_tensors, inputs_mems);
}


void execute(const OpDef& def,
        SmallVector<TensorPtr> inputs,
        SmallVector<TensorPtr> outputs,
        SmallVector<TensorPtr> workspace) {
    if (memory_forward_success(def, inputs)) {
        return;
    }
    return proxy_graph_detail::execute(def, inputs, outputs, workspace);
}

OP_TRAIT_REG(Reduce, Reduce, opr::Reduce)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_mem_desc(infer_output_mem_desc)
        .execute(execute)
        .fallback();
}  // namespace reduce
}  // namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
