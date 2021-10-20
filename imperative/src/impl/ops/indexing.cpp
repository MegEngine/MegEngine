/**
 * \file imperative/src/impl/ops/indexing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"

#include "../op_trait.h"

#include "megbrain/opr/indexing.h"

namespace mgb {
namespace imperative {

namespace {
namespace indexing_one_hot {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    auto& op = def.cast_final_safe<IndexingOneHot>();

    mgb_assert(input_descs.size() == 2, "IndexingOneHot expects two inputs");

    auto comp_node = input_descs[0].comp_node;
    TensorLayout src = input_descs[0].layout, index = input_descs[1].layout;

    mgb_assert(index.dtype == dtype::Int32(), "index dtype must be int32");

    if (!src.ndim) {
        return {{{{{}, src.dtype}, comp_node}}, false};
    }

    mgb_assert(src.ndim >= 2, "src ndim must be at least 2");
    mgb_assert(src.is_contiguous(), "src should be contiguous");
    mgb_assert(
            op.axis >= 0 && op.axis < src.ndim, "axis %d not exists in src", op.axis);

    TensorLayout dst = src;
    dst.shape[op.axis] = 1;
    dst.init_contiguous_stride();

    if (!index.ndim) {
        return {{{dst, comp_node}}, false};
    }

    mgb_assert(index.is_contiguous(), "index should be all contiguous");
    mgb_assert(
            index.eq_shape(src.remove_axis(op.axis)), "index shape doesn't match src");

    return {{{dst, comp_node}}, true};
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const IndexingOneHot&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::IndexingOneHot::make(inputs[0], inputs[1], op.param(), config);
}

OP_TRAIT_REG(IndexingOneHot, IndexingOneHot)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_var_node(apply_on_var_node)
        .fallback();

}  // namespace indexing_one_hot
}  // anonymous namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
