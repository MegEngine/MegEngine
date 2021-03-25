/**
 * \file imperative/src/impl/ops/broadcast.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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

namespace broadcast {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    node_->cast_final_safe<opr::Broadcast>();
    return Broadcast::make();
}

auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Broadcast>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 2, "Broadcast expects 2 inputs; got %lu actually", nr_inp);
    OperatorNodeConfig config{op.make_name()};
    return opr::Broadcast::make(inputs[0], inputs[1], config);
}

bool valid_broadcast(const TensorShape& src_shape,
                     const TensorShape& tar_shape) {
    size_t src_ndim = src_shape.ndim, tar_ndim = tar_shape.ndim;
    if (src_ndim > tar_ndim) {
        return false;
    }
    size_t min_ndim = src_ndim;
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

    TensorShape out_shape;
    if (tshp.layout.ndim == 0 || tshp.value.empty()) {
        out_shape.ndim = 0;
        return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, false};
    }
    mgb_assert(
        tshp.layout.ndim == 1,
        "target shape of Broadcast expects ndim=1; got ndim=%lu actually",
        tshp.layout.ndim);

    size_t target_ndim = tshp.layout.shape[0];
    out_shape.ndim = target_ndim;
    auto* ptr = tshp.value.ptr<dt_int32>();
    for (size_t i = 0; i < target_ndim; ++i) {
        out_shape[i] = ptr[i];
    }
    mgb_assert(valid_broadcast(src.layout, out_shape),
               "the input shape %s can not be broadcasted to target shape %s", 
               src.layout.to_string().c_str(),
               out_shape.to_string().c_str());

    return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, true};
}

OP_TRAIT_REG(Broadcast, Broadcast, opr::Broadcast)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .fallback();
} // broadcast

namespace reshape {

auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const Reshape&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::Reshape::make(inputs[0], inputs[1], op.param(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<Reshape>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 2, "Reshape expects 2 inputs; got %lu actually", nr_inp);
    auto&& src = inputs[0];
    auto&& tshp = inputs[1];

    TensorShape out_shape;
    if (tshp.layout.ndim == 0 || tshp.value.empty()) {
        out_shape.ndim = 0;
        return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, false};
    }
    mgb_assert(
        tshp.layout.ndim == 1,
        "target shape of Reshape expects ndim=1; got ndim=%lu actually",
        tshp.layout.ndim);

    if (src.layout.ndim == 0 && op.axis != opr::Reshape::Param::INVALID_AXIS) {
        return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, false};
    }

    size_t target_ndim = tshp.layout.shape[0];
    out_shape.ndim = target_ndim;
    auto* ptr = tshp.value.ptr<dt_int32>();
    for (size_t i = 0; i < target_ndim; ++i) {
        out_shape[i] = ptr[i];
    }

    if (src.layout.ndim == 0) {
        return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, false};
    }

    if (op.axis != opr::Reshape::Param::INVALID_AXIS) {
        mgb_assert(out_shape[op.axis] == -1);
        out_shape[op.axis] = 1;
        mgb_assert(src.layout.total_nr_elems() % out_shape.total_nr_elems() == 0,
            "can not reshape from %s to %s",
            src.layout.to_string().c_str(),
            out_shape.to_string().c_str());
        out_shape[op.axis] = src.layout.total_nr_elems() / out_shape.total_nr_elems();
    } else {
        mgb_assert(src.layout.total_nr_elems() == out_shape.total_nr_elems(),
            "can not reshape from %s to %s",
            src.layout.to_string().c_str(),
            out_shape.to_string().c_str());
    }
    return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, true};
}

OP_TRAIT_REG(Reshape, Reshape)
    .apply_on_var_node(apply_on_var_node)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .fallback();
} // reshape

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
