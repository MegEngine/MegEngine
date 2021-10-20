/**
 * \file imperative/src/impl/ops/batch_norm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/batch_norm.h"
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb {
namespace imperative {

namespace {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::BatchNorm>();
    return BatchNorm::make(node->param());
}

cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& bn_opr = def.cast_final_safe<BatchNorm>();
    size_t nr_inp = inputs.size();
    mgb_assert(
            nr_inp == 3 || nr_inp == 5,
            "BatchNorm expects 3 or 5 inputs; got %lu actually", nr_inp);
    OperatorNodeConfig config{bn_opr.make_name()};
    if (nr_inp == 3) {
        return opr::BatchNorm::make(
                       inputs[0], inputs[1], inputs[2], bn_opr.param(), config)[0]
                .node()
                ->owner_opr();
    } else {
        return opr::BatchNorm::make(
                       inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                       bn_opr.param(), config)[0]
                .node()
                ->owner_opr();
    }
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<BatchNorm>();
    size_t nr_inp = inputs.size();
    mgb_assert(
            nr_inp == 3 || nr_inp == 5,
            "BatchNorm expects 3 or 5 inputs; got %lu actually", nr_inp);
    // need running mean/variance
    bool need_stat = (nr_inp == 5) && op_def.fwd_mode == BatchNorm::FwdMode::TRAINING;
    size_t nr_out = need_stat ? 6 : 4;
    SmallVector<LogicalTensorDesc> out_shapes(nr_out);
    auto&& i0 = inputs[0];
    auto&& i1 = inputs[1];
    // [running_mean, running_var,] save_mean, save_var
    for (size_t i = 0; i < nr_out - 2; ++i) {
        out_shapes[i] = {i1.layout, i1.comp_node};
    }
    out_shapes[nr_out - 2] = {
            TensorLayout({0}, dtype::Byte()), i0.comp_node};  // reserve
    out_shapes[nr_out - 1] = {i0.layout, i0.comp_node};       // output
    return {out_shapes, out_shapes[nr_out - 1].layout.ndim != 0};
}

OP_TRAIT_REG(BatchNorm, BatchNorm, opr::BatchNorm)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .fallback();
}  // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
