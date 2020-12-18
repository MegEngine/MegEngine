/**
 * \file imperative/src/impl/ops/broadcast.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/tensor_manip.h"

#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    node_->cast_final_safe<opr::Broadcast>();
    return Broadcast::make();
}

cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    def.cast_final_safe<Broadcast>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 2, "Broadcast expects 2 inputs; got %lu actually", nr_inp);
    return opr::Broadcast::make(inputs[0], inputs[1]).node()->owner_opr();
}

bool valid_broadcast(const TensorShape& src_shape,
                     const TensorShape& tar_shape) {
    size_t src_ndim = src_shape.ndim, tar_ndim = tar_shape.ndim;
    if (src_ndim > tar_ndim) {
        return false;
    }
    size_t min_ndim = src_ndim < tar_ndim ? src_ndim : tar_ndim;
    for (size_t i = 0; i < min_ndim; ++i) {
        if (src_shape[src_ndim - i - 1] != 1 &&
            src_shape[src_ndim - i - 1] != tar_shape[tar_ndim - i - 1]) {
            return false;
        }
    }
    return true;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    def.cast_final_safe<Broadcast>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 2, "Broadcast expects 2 inputs; got %lu actually", nr_inp);
    auto&& src = inputs[0];
    auto&& tshp = inputs[1];

    TensorLayout out_layout = src.layout;
    if (tshp.layout.ndim == 0 || tshp.value.empty()) {
        out_layout.ndim = 0;
        return {{{out_layout, src.comp_node}}, true};
    }
    mgb_assert(
        tshp.layout.ndim == 1, 
        "target shape of Broadcast expects ndim=1; got ndim=%lu actually", 
        tshp.layout.ndim);

    size_t target_ndim = tshp.layout.shape[0];
    out_layout.ndim = target_ndim;
    auto* ptr = tshp.value.ptr<dt_int32>();
    for(size_t i=0; i<target_ndim; ++i) {
        out_layout.shape[i] = ptr[i];
    }
    mgb_assert(valid_broadcast(src.layout, out_layout),
               "the input shape %s can not be broadcasted to target shape %s", 
               src.layout.TensorShape::to_string().c_str(),
               out_layout.TensorShape::to_string().c_str());

    return {{{out_layout, src.comp_node}}, true};
}

OP_TRAIT_REG(Broadcast, Broadcast, opr::Broadcast)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .fallback();
} // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
