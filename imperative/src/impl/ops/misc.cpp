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
SymbolVar apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<CheckNonFinite>();
    OperatorNodeConfig config{op.make_name()};
    return opr::CheckNonFinite::make(inputs, {}, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    size_t size = inputs.size();

    auto dest = Tensor::make(
            TensorLayout(TensorShape({1}), dtype::Int32()), inputs[0]->comp_node());
    auto cn = dest->comp_node();
    auto&& dnn_opr = opr::intl::create_megdnn_opr<megdnn::CheckNonFinite>(cn);
    size_t wk_size = 0;
    SmallVector<megdnn::TensorND> srcs(size);
    for (size_t i = 0; i < size; ++i) {
        srcs[i] = inputs[i]->dev_tensor().as_megdnn();
    }
    wk_size = dnn_opr->get_workspace_in_bytes(srcs, dest->layout());
    auto wk = Blob::make(cn, wk_size);
    megdnn::Workspace dnn_wk(wk->storage().get(), wk_size);
    dnn_opr->exec(srcs, dest->dev_tensor().as_megdnn(), dnn_wk);
    return {dest};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    SmallVector<LogicalTensorDesc> dests(1);
    dests[0].comp_node = inputs[0].comp_node;
    dests[0].layout = TensorLayout(TensorShape({1}), dtype::Int32());
    return {dests, true};
}
SmallVector<LogicalTensorDesc> infer_output_attrs(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<LogicalTensorDesc> dests(1);
    dests[0].comp_node = inputs[0]->comp_node();
    dests[0].layout = TensorLayout(TensorShape({1}), dtype::Int32());
    return dests;
}
std::tuple<SmallVector<MemoryDesc>, SmallVector<MemoryDesc>> infer_output_mem_desc(
        const OpDef& def, const SmallVector<TensorPtr>& inputs_tensors,
        const SmallVector<MemoryDesc>& inputs_mems) {
    return {{}, {}};
}
OP_TRAIT_REG(CheckNonFinite, CheckNonFinite)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .infer_output_mem_desc(infer_output_mem_desc)
        .fallback();
}  // namespace check_non_finite

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
