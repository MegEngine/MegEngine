/**
 * \file imperative/src/impl/ops/tensor_manip.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "../op_trait.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/misc.h"

namespace mgb {
namespace imperative {

namespace check_non_finite {
SymbolVarArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<CheckNonFinite>();
    OperatorNodeConfig config{op.make_name()};
    return opr::CheckNonFinite::make(inputs, op.param(), config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    size_t size = inputs.size();
    auto&& op = def.cast_final_safe<CheckNonFinite>();
    SmallVector<TensorPtr> outputs(size + 1);
    outputs[size] = Tensor::make(
            TensorLayout(TensorShape({1}), dtype::Int32()), inputs[0]->comp_node());

    auto dest = outputs[size];
    auto cn = dest->comp_node();
    auto&& dnn_opr = opr::intl::create_megdnn_opr<megdnn::CheckNonFinite>(cn);
    size_t wk_size = 0;
    SmallVector<megdnn::TensorND> srcs(size);
    // copy an outputs to the dnn for inplace
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = Tensor::make(inputs[i]->layout(), inputs[0]->comp_node());
        outputs[i]->dev_tensor().copy_from_fixlayout(inputs[i]->dev_tensor());
        srcs[i] = outputs[i]->dev_tensor().as_megdnn();
    }
    megdnn::CheckNonFinite::Param param({op.scale});
    dnn_opr->param() = param;
    wk_size = dnn_opr->get_workspace_in_bytes(srcs, dest->layout());
    auto wk = Blob::make(cn, wk_size);
    megdnn::Workspace dnn_wk(wk->storage().get(), wk_size);
    dnn_opr->exec(srcs, dest->dev_tensor().as_megdnn(), dnn_wk);
    return outputs;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    size_t size = inputs.size();
    SmallVector<LogicalTensorDesc> dests(size + 1);
    for (size_t i = 0; i < size; ++i) {
        dests[i].comp_node = inputs[i].comp_node;
        dests[i].layout = inputs[i].layout;
    }
    dests[size].comp_node = inputs[0].comp_node;
    dests[size].layout = TensorLayout(TensorShape({1}), dtype::Int32());
    return {dests, true};
}
SmallVector<LogicalTensorDesc> infer_output_attrs(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    size_t size = inputs.size();
    SmallVector<LogicalTensorDesc> dests(size + 1);
    for (size_t i = 0; i < size; ++i) {
        dests[i].comp_node = inputs[i]->comp_node();
        dests[i].layout = inputs[i]->layout();
    }
    dests[size].comp_node = inputs[0]->comp_node();
    dests[size].layout = TensorLayout(TensorShape({1}), dtype::Int32());
    return dests;
}

OP_TRAIT_REG(CheckNonFinite, CheckNonFinite)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .fallback();
}  // namespace check_non_finite

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
