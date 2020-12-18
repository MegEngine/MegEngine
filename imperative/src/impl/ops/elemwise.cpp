/**
 * \file imperative/src/impl/ops/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/basic_arith.h"

#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Elemwise>();
    return Elemwise::make(node->param().mode);
}

cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& elemwise_opr = def.cast_final_safe<Elemwise>();
    return opr::Elemwise::make(inputs, elemwise_opr.mode).node()->owner_opr();
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Elemwise>();
    auto trait = megdnn::Elemwise::ModeTrait::from_mode(op_def.mode);
    mgb_assert(inputs.size() == trait.arity,
               "%s expects %u inputs; got %zu actually", trait.name,
               trait.arity, inputs.size());
    TensorShapeArray inp_shapes;
    DType out_dt;
    CompNode out_cn;
    for (size_t i = 0; i < inputs.size(); ++ i) {
        auto &&t = inputs[i];
        if (!i) {
            out_cn = t.comp_node;
            out_dt = t.layout.dtype;
        } else {
            mgb_assert(t.comp_node == out_cn);
            mgb_assert(t.layout.dtype == out_dt);
        }
        if (t.layout.ndim > 0) {
            inp_shapes.push_back(t.layout);
        } else {
            TensorLayout out_layout;
            out_layout.ndim = 0;
            out_layout.dtype = out_dt;
            return {{{out_layout, out_cn}}, true};
        }
    }

    auto&& out_shape = opr::Elemwise::get_output_var_shape(op_def.mode, inp_shapes);
    return {{{TensorLayout(out_shape, out_dt, inputs[0].layout.format), out_cn}}, true};
}

OP_TRAIT_REG(Elemwise, Elemwise, opr::Elemwise)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .fallback();
} // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
