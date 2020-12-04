/**
 * \file imperative/src/impl/ops/tensor_manip.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/opr/tensor_manip.h"
#include "../op_trait.h"

namespace mgb::imperative {
namespace {

cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();
    return opr::GetVarShape::make(inputs, op_def.param()).node()->owner_opr();
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();
    mgb_assert(inputs.size() == 1, "GetVarShape take 1 input, got %lu", inputs.size());
    auto&& inp = inputs[0];
    auto&& shp = inp->layout();
    mgb_assert(shp.ndim != 0, "input shape invalid");
    HostTensorND hv;
    if (op_def.axis == opr::GetVarShape::Param::INVALID_AXIS){
        hv = HostTensorND(inp->comp_node(), {shp.ndim}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        for (size_t i = 0; i < shp.ndim; ++i) {
            ptr[i] = shp.shape[i];
        }
    }else{
        mgb_assert(op_def.axis < shp.ndim);
        hv = HostTensorND(inp->comp_node(), {1}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = shp.shape[op_def.axis];
    }
    return {Tensor::make(std::move(hv))};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();
    mgb_assert(inputs.size() == 1, "GetVarShape take 1 input, got %lu", inputs.size());
    auto&& desc = inputs[0];
    if (!desc.layout.ndim) {
        return {{{TensorLayout(dtype::Int32()), desc.comp_node}}, true};
    }
    DeviceTensorND value;
    if (op_def.axis == opr::GetVarShape::Param::INVALID_AXIS){
        value = DeviceTensorND(CompNode::default_cpu(), {desc.layout.ndim}, dtype::Int32());
        auto* ptr = value.ptr<dt_int32>();
        for (size_t i = 0; i < desc.layout.ndim; ++i) {
            ptr[i] = desc.layout[i];
        }
    }else{
        mgb_assert(op_def.axis < desc.layout.ndim);
        value = DeviceTensorND(CompNode::default_cpu(), {1}, dtype::Int32());
        auto* ptr = value.ptr<dt_int32>();
        ptr[0] = desc.layout[op_def.axis];
    }
    return {{{value.layout(), desc.comp_node, std::move(value)}}, true};
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::GetVarShape>();
    return GetVarShape::make(node->param());
}

OP_TRAIT_REG(GetVarShape, GetVarShape, opr::GetVarShape)
    .make_from_op_node(make_from_op_node)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .apply_on_var_node(apply_on_var_node)
    .apply_on_physical_tensor(apply_on_physical_tensor)
    .fallback();

TensorShapeArray get_shapes(const std::vector<std::vector<size_t>>& shapes) {
    TensorShapeArray ret;
    for (auto&& i:shapes) {
        SmallVector<size_t> shape(i.begin(), i.end());
        TensorShape shp(shape);
        ret.push_back(shp);
    }
    return ret;
}

cg::OperatorNodeBase* param_pack_split_apply_on_var_node(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& param = def.cast_final_safe<ParamPackSplit>();
    auto&& graph = inputs[0]->owner_graph();

    auto&& shapes = get_shapes(param.shapes);
    cg::OperatorNodeConfig config;
    cg::OperatorNodeBase* opr =
            graph->insert_opr(std::make_unique<mgb::opr::ParamPackSplit>(
                    inputs[0], param.offsets, shapes, config));
    return opr;
}

SmallVector<TensorPtr> param_pack_split_apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto&& param = def.cast_final_safe<ParamPackSplit>();
    mgb_assert(inputs.size() == 1, "ParamPackSplit take 1 input, got %lu", inputs.size());
    auto&& inp = inputs[0];
    auto&& shp = inp->layout();
    mgb_assert(shp.ndim == 1, "ParamPackSplit input shape invalid, ndim should be 1");
    mgb_assert(param.shapes.size() * 2 == param.offsets.size());
    SmallVector<TensorPtr> ret;
    auto&& shapes = get_shapes(param.shapes);
    size_t dtype_size = inputs[0]->layout().dtype.size();
    for (size_t i = 0; i < shapes.size(); ++i) {
        ret.push_back(
                inputs[0]->sub(param.offsets[i * 2] * dtype_size, shapes[i]));
    }
    return ret;
}

OP_TRAIT_REG(ParamPackSplit, ParamPackSplit, mgb::opr::ParamPackSplit)
        .apply_on_var_node(param_pack_split_apply_on_var_node)
        .apply_on_physical_tensor(param_pack_split_apply_on_physical_tensor)
        .fallback();

cg::OperatorNodeBase* param_pack_concat_apply_on_var_node(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& param = def.cast_final_safe<ParamPackConcat>();
    auto&& graph = inputs[0]->owner_graph();

    VarNodeArray inps(inputs.begin(), inputs.end() - 1);
    cg::OperatorNodeConfig config;
    cg::OperatorNodeBase* opr =
            graph->insert_opr(std::make_unique<mgb::opr::ParamPackConcat>(
                    inps, inputs.back(), param.offsets, config));
    return opr;
}

OP_TRAIT_REG(ParamPackConcat, ParamPackConcat, mgb::opr::ParamPackConcat)
        .apply_on_var_node(param_pack_concat_apply_on_var_node)
        .fallback();
} // namespace

} // namespace mgb::imperative
