/**
 * \file imperative/src/impl/ops/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/stats.h"
#include "megbrain/opr/utility.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "../algo_chooser.h"
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb::imperative {

namespace {
namespace pooling {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& pool = static_cast<const Pooling&>(def);
    OperatorNodeConfig config{pool.make_name()};
    return opr::Pooling::make(inputs[0], pool.param(), pool.policy(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(
            inputs.size() == 1, "num of inputs of pooling should be 1 but you give %zu",
            inputs.size());

    auto&& op_def = def.cast_final_safe<Pooling>();
    auto&& inp = inputs[0];
    auto& inp_cn = inp.comp_node;

    if (inp.layout.ndim == 0) {
        return {{{TensorLayout{inp.layout.dtype}, inp_cn, {}}}, false};
    }

    TensorLayout oup_layout;
    megdnn::Pooling::deduce_layout_impl(inp.layout, op_def.param(), oup_layout);

    return {{{oup_layout, inp_cn, {}}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    mgb_assert(
            inputs.size() == 1, "num of inputs of pooling should be 1 but you give %zu",
            inputs.size());

    auto&& op_def = def.cast_final_safe<Pooling>();
    auto cn = inputs[0]->comp_node();
    megdnn::TensorND inp_tensornd = inputs[0]->dnn_tensor();

    DnnOprCaller<megdnn::Pooling> caller(cn);
    auto&& dnn_opr = caller.op;
    dnn_opr->param() = op_def.param();

    TensorLayout& oup_layout = output_descs[0].layout;
    if (!validated) {
        megdnn::Pooling::deduce_layout_impl(
                inp_tensornd.layout, op_def.param(), oup_layout);
    }
    DeviceTensorND out_devtensor =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, oup_layout);

    size_t wk_size = setup_algo<megdnn::Pooling>(
            {inp_tensornd.layout, oup_layout}, dnn_opr.get(), 0, false, false, cn,
            op_def.policy(), false);

    megdnn::Workspace dnn_wk;
    if (wk_size != 0) {
        auto wk = Blob::make(cn, wk_size);
        dnn_wk.raw_ptr = wk->storage().get();
        dnn_wk.size = wk_size;
    }

    dnn_opr->exec(inp_tensornd, out_devtensor.as_megdnn(), {});
    return {Tensor::make(out_devtensor)};
}

OP_TRAIT_REG(Pooling, Pooling)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace pooling
}  // namespace

}  // namespace mgb::imperative
