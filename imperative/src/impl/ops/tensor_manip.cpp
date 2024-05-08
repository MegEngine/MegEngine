#include "megbrain/opr/tensor_manip.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/opr_attr.h"

#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb::imperative {

namespace get_var_shape {
cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();
    OperatorNodeConfig config{op_def.make_name()};
    return opr::GetVarShape::make(inputs, op_def.param(), config).node()->owner_opr();
}

DispatchMode decide_dispatch_mode(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
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
        const OpDef& def, const SmallVector<DeviceTensorND>& inputs,
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
    mgb_assert(
            (*outputs)[0].comp_node() == CompNode::default_cpu(),
            "GetVarShape's apply_on_device_tensornd should receive default_cpu "
            "outputs.");

    HostTensorND hv;
    if (op_def.axis == opr::GetVarShape::Param::INVALID_AXIS) {
        hv = HostTensorND(CompNode::default_cpu(), {shp.ndim}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        for (size_t i = 0; i < shp.ndim; ++i) {
            ptr[i] = shp.shape[i];
        }
    } else {
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

HostTensorND get_var_shape_host_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<DeviceTensorND> input_tensornds;
    for (auto&& inp : inputs) {
        input_tensornds.push_back(inp->dev_tensor(false));
    }
    SmallVector<DeviceTensorND> output_tensornds = {
            {CompNode::default_cpu(), dtype::Int32()}};
    apply_on_device_tensornd(def, input_tensornds, &output_tensornds);
    // restore to input comp_node
    return HostTensorND::make_proxy(output_tensornds[0])
            .proxy_to_comp_node(inputs[0]->comp_node());
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    return {Tensor::make(get_var_shape_host_tensor(def, inputs))};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<GetVarShape>();
    auto&& desc = inputs[0];
    TensorShape shp;
    if (inputs.size() == 1) {
        shp = desc.layout;
    } else {
        TensorShapeArray src(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            src[i] = inputs[i].layout;
            if (!src[i].ndim) {
                return {{{TensorLayout(dtype::Int32()), desc.comp_node}}, false};
            }
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
    } else {
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
}  // namespace get_var_shape

namespace param_pack {
TensorShapeArray get_shapes(const std::vector<std::vector<size_t>>& shapes) {
    TensorShapeArray ret;
    for (auto&& i : shapes) {
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
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& param = def.cast_final_safe<ParamPackSplit>();
    mgb_assert(
            inputs.size() == 1, "ParamPackSplit take 1 input, got %lu", inputs.size());
    auto&& inp = inputs[0];
    auto&& shp = inp->layout();
    mgb_assert(shp.ndim == 1, "ParamPackSplit input shape invalid, ndim should be 1");
    mgb_assert(param.shapes.size() * 2 == param.offsets.size());
    SmallVector<TensorPtr> ret;
    auto&& shapes = get_shapes(param.shapes);
    size_t dtype_size = inputs[0]->layout().dtype.size();
    for (size_t i = 0; i < shapes.size(); ++i) {
        // memory forward
        ret.push_back(inputs[0]->sub(param.offsets[i * 2] * dtype_size, shapes[i]));
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
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& param = def.cast_final_safe<ParamPackConcat>();

    mgb_assert(inputs.size() > 1, "param_pack should have at least one input");
    auto comp_node = inputs.front()->comp_node();
    auto dtype = inputs.front()->dtype();
    size_t nr_inputs = inputs.size() - 1;
    size_t nr_elems = 0;
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto& input = inputs[i];
        mgb_assert(
                comp_node == input->comp_node(),
                "inputs for param_pack_concat must in same comp_node");
        mgb_assert(
                dtype == input->dtype(),
                "inputs for param_pack_concat must have same dtype");
        nr_elems += input->layout().total_nr_elems();
    }
    auto dest_layout = TensorLayout({nr_elems}, dtype);
    auto output = Tensor::make(dest_layout, comp_node);
    // FIXME: add param to ParamPackConcat
    DnnOprCaller<megdnn::ParamPackConcat> caller{comp_node};
    HostTensorStorage srcs_storage{comp_node};
    srcs_storage.ensure_size(sizeof(void*) * nr_inputs);
    TensorLayout srcs_layout = TensorLayout{{nr_inputs}, dtype::Int32()};
    HostTensorND srcs_tensornd;
    srcs_tensornd.reset(srcs_storage, srcs_layout);
    auto* srcs_raw_ptr = reinterpret_cast<void**>(srcs_storage.ptr());
    for (size_t i = 0; i < nr_inputs; ++i) {
        srcs_raw_ptr[i] = inputs[i]->dnn_tensor().raw_ptr();
    }

    // hack
    if (comp_node.device_type() == CompNode::DeviceType::ATLAS) {
        HostTensorStorage offsets_storage{comp_node};
        offsets_storage.ensure_size(sizeof(int32_t) * param.offsets.size());
        HostTensorND offsets_tensornd;
        offsets_tensornd.reset(offsets_storage, inputs.back()->layout());
        auto* offsets_raw_ptr = reinterpret_cast<int32_t*>(offsets_storage.ptr());
        for (size_t i = 0; i < param.offsets.size(); i++) {
            offsets_raw_ptr[i] = param.offsets[i];
        }
        caller.exec_with_ws(
                srcs_tensornd.as_megdnn(), offsets_tensornd.as_megdnn(), output);
        async_release(offsets_tensornd);
    } else {
        caller.exec_with_ws(srcs_tensornd.as_megdnn(), inputs.back(), output);
    }

    async_release(srcs_tensornd);
    return {output};
}

OP_TRAIT_REG(ParamPackConcat, ParamPackConcat, mgb::opr::ParamPackConcat)
        .apply_on_var_node(param_pack_concat_apply_on_var_node)
        .apply_on_physical_tensor(param_pack_concat_apply_on_physical_tensor)
        .fallback();
}  // namespace param_pack

namespace split {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    using Options = opr::Split::Options;
    auto* node = &node_->cast_final_safe<opr::Split>();
    auto&& opt = node->options();
    int axis = opt.axis;
    mgb_assert(
            opt.method == Options::Method::SPECIFY,
            "only Split with SPECIFY output shapes is supported");
    mgb_assert(opt.partition.size() == opt.nr_part);
    return Split::make(axis, 0);
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    using Options = opr::Split::Options;
    auto&& sp = static_cast<const Split&>(def);
    OperatorNodeConfig config{sp.make_name()};
    opr::Split::Options opt;
    if (sp.nsections) {
        opt = Options::make_average(sp.axis, sp.nsections);
        opt.method = Options::Method::CALL_BACK;
    } else {
        opt.axis = sp.axis;
        opt.method = Options::Method::SPECIFY;
        mgb_assert(inputs.size() > 1);
        opt.nr_part = inputs.size() - 1;
        opt.partition.resize(opt.nr_part);
        for (size_t i = 1; i < inputs.size(); ++i)
            opt.partition[i - 1] = inputs[i];
    }
    return opr::Split::make(inputs[0], opt, config);
}

OP_TRAIT_REG(Split, Split, opr::Split)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .fallback();

}  // namespace split

namespace masked_fill {

cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op_def = def.cast_final_safe<MaskedFill>();
    OperatorNodeConfig config{op_def.make_name()};
    mgb_assert(inputs.size() == 2);
    return opr::MaskedFill::make(inputs[0], inputs[1], op_def.param(), config)
            .node()
            ->owner_opr();
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[1] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    return {{{{input_descs[0].layout, input_descs[0].layout.dtype},
              input_descs[0].comp_node}},
            input_descs[0].layout.ndim != 0};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<MaskedFill>();
    auto&& inp = inputs[0];
    auto&& mask = inputs[1];

    TensorLayout outlayout(inp->layout(), inp->layout().dtype);

    auto output = Tensor::make(outlayout, inp->comp_node());

    DnnOprCaller<megdnn::MaskedFill> dnn_opr{inp->comp_node(), op.param()};
    dnn_opr.exec_with_ws(inp, mask, output);
    return {output};
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::MaskedFill>();
    return MaskedFill::make(node->param());
}

OP_TRAIT_REG(MaskedFill, MaskedFill, mgb::opr::MaskedFill)
        .get_input_layout_constraint(get_input_layout_constraint)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .apply_on_var_node(apply_on_var_node)
        .make_from_op_node(make_from_op_node)
        .fallback();

}  // namespace masked_fill

}  // namespace mgb::imperative
