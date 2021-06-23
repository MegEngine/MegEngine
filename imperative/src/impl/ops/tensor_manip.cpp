/**
 * \file imperative/src/impl/ops/tensor_manip.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/opr/tensor_manip.h"

#include "../async_releaser.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb::imperative {

namespace get_var_shape {
cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();
    OperatorNodeConfig config{op_def.make_name()};
    return opr::GetVarShape::make(inputs, op_def.param(), config).node()->owner_opr();
}

DispatchMode decide_dispatch_mode(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    bool host_computable = true;
    for (auto&& inp : inputs) {
        // FIXME(czh): remove value check after proxy graph's
        // apply_on_device_tensornd is supported and output Tensor
        // is made before add_task.
        // then if layout is valid, ptr->layout must be ready
        if (inp.value.empty() || inp.value.layout().ndim == 0) {
            host_computable = false;
            break;
        }
    }
    return host_computable ? DEFAULT_CPU : KERNEL;
}

void apply_on_device_tensornd(
        const OpDef& def,
        const SmallVector<DeviceTensorND>& inputs,
        SmallVector<DeviceTensorND>* outputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();

    TensorShape shp;
    if (inputs.size() == 1) {
        shp = inputs[0].layout();
    } else {
        TensorShapeArray src(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            src[i] = inputs[i].layout();
        }
        megdnn::Elemwise::deduce_shape(src, shp);
    }

    mgb_assert(shp.ndim != 0, "input shape invalid");
    mgb_assert((*outputs)[0].comp_node() == CompNode::default_cpu(),
        "GetVarShape's apply_on_device_tensornd should receive default_cpu outputs.");

    HostTensorND hv;
    if (op_def.axis == opr::GetVarShape::Param::INVALID_AXIS) {
        hv = HostTensorND(CompNode::default_cpu(), {shp.ndim}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        for (size_t i = 0; i < shp.ndim; ++i) {
            ptr[i] = shp.shape[i];
        }
    }else{
        int32_t axis = op_def.axis;
        if (axis < 0) {
            axis += shp.ndim;
        }
        mgb_assert(axis >= 0 && axis < (int32_t)shp.ndim);
        hv = HostTensorND(CompNode::default_cpu(), {1}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = shp.shape[axis];
    }
    (*outputs)[0] = DeviceTensorND::make_proxy(hv);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    SmallVector<DeviceTensorND> input_tensornds;
    input_tensornds.reserve(inputs.size());
    for (auto&& inp : inputs) {
        input_tensornds.push_back(inp->dev_tensor());
    }
    SmallVector<DeviceTensorND> output_tensornds = {{CompNode::default_cpu(), dtype::Int32()}};

    apply_on_device_tensornd(def, input_tensornds, &output_tensornds);

    // restore to input comp_node
    HostTensorND host_tensornd = HostTensorND::make_proxy(output_tensornds[0])
        .proxy_to_comp_node(inputs[0]->comp_node());
    return {Tensor::make(std::move(host_tensornd))};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();
    auto&& desc = inputs[0];
    TensorShape shp;
    if (inputs.size() == 1) {
        shp = desc.layout;
    } else {
        TensorShapeArray src(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            src[i] = inputs[i].layout;
        }
        megdnn::Elemwise::deduce_shape(src, shp);
    }
    if (!shp.ndim) {
        return {{{TensorLayout(dtype::Int32()), desc.comp_node}}, false};
    }
    DeviceTensorND value;
    if (op_def.axis == opr::GetVarShape::Param::INVALID_AXIS) {
        value = DeviceTensorND(CompNode::default_cpu(), {shp.ndim}, dtype::Int32());
        auto* ptr = value.ptr<dt_int32>();
        for (size_t i = 0; i < shp.ndim; ++i) {
            ptr[i] = shp[i];
        }
    }else{
        int32_t axis = op_def.axis;
        if (axis < 0) {
            axis += shp.ndim;
        }
        mgb_assert(axis >= 0 && axis < (int32_t)shp.ndim);
        value = DeviceTensorND(CompNode::default_cpu(), {1}, dtype::Int32());
        auto* ptr = value.ptr<dt_int32>();
        ptr[0] = shp[axis];
    }
    return {{{value.layout(), desc.comp_node, std::move(value)}}, true};
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::GetVarShape>();
    return GetVarShape::make(node->param());
}

OP_TRAIT_REG(GetVarShape, GetVarShape, opr::GetVarShape)
    .make_from_op_node(make_from_op_node)
    .decide_dispatch_mode(decide_dispatch_mode)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .apply_on_var_node(apply_on_var_node)
    .apply_on_device_tensornd(apply_on_device_tensornd)
    .apply_on_physical_tensor(apply_on_physical_tensor)
    .fallback();
} // get_var_shape

namespace param_pack {
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
    OperatorNodeConfig config(param.make_name());
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
        // memory forward
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
    OperatorNodeConfig config{param.make_name()};
    cg::OperatorNodeBase* opr =
            graph->insert_opr(std::make_unique<mgb::opr::ParamPackConcat>(
                    inps, inputs.back(), param.offsets, config));
    return opr;
}

SmallVector<TensorPtr> param_pack_concat_apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    def.cast_final_safe<ParamPackConcat>();
    mgb_assert(inputs.size() > 1, "param_pack should have at least one input");
    auto comp_node = inputs.front()->comp_node();
    auto dtype = inputs.front()->dtype();
    size_t nr_inputs = inputs.size() - 1;
    size_t nr_elems = 0;
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto& input = inputs[i];
        mgb_assert(comp_node == input->comp_node(), "inputs for param_pack_concat must in same comp_node");
        mgb_assert(dtype == input->dtype(), "inputs for param_pack_concat must have same dtype");
        nr_elems += input->layout().total_nr_elems();
    }
    auto dest_layout = TensorLayout({nr_elems}, dtype);
    auto output = Tensor::make(dest_layout, comp_node);
    auto caller = DnnOprCaller<megdnn::ParamPackConcat>(comp_node);
    size_t srcs_size = sizeof(void*)*nr_inputs;
    void** srcs_raw_ptr = (void**)comp_node.alloc_host(srcs_size);
    std::shared_ptr<dt_byte> srcs_ptr = {(dt_byte*)srcs_raw_ptr, [comp_node](dt_byte* ptr){
        comp_node.free_host(ptr);
    }};
    TensorLayout srcs_layout = TensorLayout{{nr_inputs}, dtype::Int32()};
    size_t ws_size;
    {
        TensorShapeArray src_shapes;
        for (size_t i = 0; i < nr_inputs; ++i) {
            src_shapes.push_back(inputs[i]->shape());
        }
        ws_size = caller.op->get_workspace_in_bytes(src_shapes, inputs.back()->shape(), TensorShape{});
    }
    for (size_t i = 0; i < nr_inputs; ++i) {
        srcs_raw_ptr[i] = inputs[i]->dev_tensor().as_megdnn().raw_ptr;
    }
    HostTensorStorage srcs_storage;
    srcs_storage.reset(comp_node, srcs_size, srcs_ptr);
    caller.op->exec({srcs_raw_ptr, srcs_layout}, inputs.back()->dev_tensor().as_megdnn(), output->dev_tensor().as_megdnn(),
            caller.create_workspace({{ws_size}, dtype::Byte()}));
    AsyncReleaser::inst()->add(HostTensorND{comp_node, srcs_layout}.storage(srcs_storage));
    return { output };
}

OP_TRAIT_REG(ParamPackConcat, ParamPackConcat, mgb::opr::ParamPackConcat)
        .apply_on_var_node(param_pack_concat_apply_on_var_node)
        .apply_on_physical_tensor(param_pack_concat_apply_on_physical_tensor)
        .fallback();
} // param_pack

} // namespace mgb::imperative
