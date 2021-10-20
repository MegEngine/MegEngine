/**
 * \file imperative/src/impl/ops/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/utility.h"

#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Elemwise>();
    return Elemwise::make(node->param().mode);
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& elemwise_opr = def.cast_final_safe<Elemwise>();
    OperatorNodeConfig config{elemwise_opr.make_name()};
    return opr::Elemwise::make(inputs, elemwise_opr.mode, config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Elemwise>();
    auto trait = megdnn::Elemwise::ModeTrait::from_mode(op_def.mode);
    mgb_assert(
            inputs.size() == trait.arity, "%s expects %u inputs; got %zu actually",
            trait.name, trait.arity, inputs.size());
    TensorShapeArray inp_shapes;
    DType out_dt;
    CompNode out_cn;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto&& t = inputs[i];
        if (!i) {
            out_cn = t.comp_node;
            out_dt = t.layout.dtype;
        } else {
            mgb_assert(t.comp_node == out_cn);
            mgb_assert(t.layout.dtype == out_dt);
        }
        if (t.layout.ndim > 0) {
            inp_shapes.push_back(t.layout);
        } else {
            TensorLayout out_layout;
            out_layout.ndim = 0;
            out_layout.dtype = out_dt;
            return {{{out_layout, out_cn}}, false};
        }
    }
    // copy from megdnn::ElemwiseForward::check_dtype
    switch (out_dt.category()) {
        case DTypeCategory::FLOAT:
            mgb_assert(trait.allow_float, "unsupport mode %s for float\n", trait.name);
            break;
        case DTypeCategory::INT:
            mgb_assert(trait.allow_int, "unsupport mode %s for int\n", trait.name);
            break;
        case DTypeCategory::BOOL:
            mgb_assert(trait.allow_bool, "unsupport mode %s for bool\n", trait.name);
            break;
        default:
            // Quantized Dtype could also be handled by this op,
            // but scales need to be the same.
            break;
    }

    auto&& out_shape = opr::Elemwise::get_output_var_shape(op_def.mode, inp_shapes);
    return {{{TensorLayout(out_shape, out_dt, inputs[0].layout.format), out_cn}}, true};
}

DispatchMode decide_dispatch_mode(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    bool host_computable = true;
    constexpr int size_threshhold = TensorShape::MAX_NDIM;
    for (auto&& inp : inputs) {
        if (inp.value.empty() || inp.value.layout().ndim == 0 ||
            inp.value.layout().total_nr_elems() > size_threshhold) {
            host_computable = false;
            break;
        }
    }
    return host_computable ? DEFAULT_CPU : KERNEL;
}

void apply_on_device_tensornd(
        const OpDef& def, const SmallVector<DeviceTensorND>& inputs,
        SmallVector<DeviceTensorND>* outputs) {
    auto&& op_def = def.cast_final_safe<Elemwise>();
    auto trait = megdnn::Elemwise::ModeTrait::from_mode(op_def.mode);
    mgb_assert(
            inputs.size() == trait.arity, "%s expects %u inputs; got %zu actually",
            trait.name, trait.arity, inputs.size());
    auto&& dnn_opr =
            opr::intl::create_megdnn_opr<megdnn::Elemwise>(inputs[0].comp_node());
    opr::Elemwise::perform(op_def.mode, (*outputs)[0], inputs, dnn_opr);
}

void execute(
        const OpDef& def, SmallVector<TensorPtr> inputs, SmallVector<TensorPtr> outputs,
        SmallVector<TensorPtr> workspace) {
    mgb_assert(outputs.size() == 1);
    SmallVector<DeviceTensorND> inp_tensornds(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inp_tensornds[i] = inputs[i]->dev_tensor();
    }
    SmallVector<DeviceTensorND> out_tensornds = {outputs[0]->dev_tensor()};
    apply_on_device_tensornd(def, inp_tensornds, &out_tensornds);
}

std::tuple<SmallVector<MemoryDesc>, SmallVector<MemoryDesc>> infer_output_mem_desc(
        const OpDef& def, const SmallVector<TensorPtr>& inputs_tensors,
        const SmallVector<MemoryDesc>& inputs_mems) {
    auto&& op_def = def.cast_final_safe<Elemwise>();
    TensorShapeArray inp_shapes(inputs_tensors.size());
    for (size_t i = 0; i < inputs_tensors.size(); ++i) {
        inp_shapes[i] = inputs_tensors[i]->layout();
    }
    TensorShape shape = opr::Elemwise::get_output_var_shape(op_def.mode, inp_shapes);
    SmallVector<MemoryDesc> outputs = {
            {{shape, inputs_tensors[0]->dtype()},
             0,
             inputs_tensors[0]->comp_node(),
             StorageIdentifier::make(1)}};
    return {outputs, {}};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    auto&& op_def = def.cast_final_safe<Elemwise>();
    SmallVector<DeviceTensorND> inp_tensornds(inputs.size());
    TensorShapeArray inp_shapes(inputs.size());
    for (unsigned i = 0; i < inputs.size(); ++i) {
        inp_tensornds[i] = inputs[i]->dev_tensor();
        inp_shapes[i] = inputs[i]->layout();
    }
    TensorShape shape = opr::Elemwise::get_output_var_shape(op_def.mode, inp_shapes);
    DeviceTensorND out = BlobManager::inst()->alloc_workspace_with_defrag(
            inp_tensornds[0].comp_node(), {shape, inp_tensornds[0].layout().dtype});
    SmallVector<DeviceTensorND> oup_tensornds = {out};
    apply_on_device_tensornd(def, inp_tensornds, &oup_tensornds);
    return {Tensor::make(oup_tensornds[0])};
}

MGB_DEFINE_OPR_CLASS(
        ForceInplaceElemwise,
        cg::SingleCNOperatorNodeBaseT<opr::mixin::MegDNNOprHolder>)  //{
public:
struct Param {
    using Mode = megdnn::Elemwise::Param::Mode;
    Mode mode;
    size_t inplace_index;
};
using Mode = Param::Mode;
ForceInplaceElemwise(
        const VarNodeArray& inputs, Param param, OperatorNodeConfig config = {})
        : Super(inputs[0]->owner_graph(), config, "device_add_update", inputs),
          m_param{param} {
    for (auto* input : inputs) {
        add_input({input});
    }
    add_output(None)
            ->set_fwd_in2out_writable_force(input(param.inplace_index))
            .add_flag(VarNode::Flag::NO_MEM_RECLAIM);
}
static SymbolVar make(const VarNodeArray& inputs, Param param) {
    return SymbolVar{inputs[0]}.insert_single_output_opr<ForceInplaceElemwise>(
            inputs, param);
}
static cg::OperatorNodeBase* shallow_copy(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config);

protected:
NodeProp* do_make_node_prop() const override {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::FORCE_UPDATE_INPUT_VAR);
    return ret;
}
void create_megdnn_opr() override {
    auto opr = DnnOprCaller<megdnn::Elemwise>::create_operator(comp_node());
    opr->param().mode = m_param.mode;
    set_megdnn_opr(std::move(opr));
}
void scn_do_execute() override {
    auto to_dnnnd = [&](auto* var) { return var->dev_tensor().as_megdnn(); };
    megdnn::TensorNDArray inputs_dnnnd;
    for (auto* input : input()) {
        inputs_dnnnd.push_back(to_dnnnd(input));
    }
    mgb_assert(
            input(m_param.inplace_index)->contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC),
            "ForceInplaceElemwise cannot be applied in internal tensor");
    auto* out_dest = output(0);
    auto* opr = static_cast<megdnn::Elemwise*>(megdnn_opr());
    opr->exec(std::move(inputs_dnnnd), to_dnnnd(out_dest));
}
void init_output_static_infer_desc() override {
    using namespace cg::static_infer;

    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), ShapeInferDesc::make_identity(input(m_param.inplace_index)));
}

private:
Param m_param;
void record_execute_deps(ExecDependencyArray& deps) override {
    record_megdnn_opr(deps);
}
};

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ForceInplaceElemwise);

cg::OperatorNodeBase* ForceInplaceElemwise::shallow_copy(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<ForceInplaceElemwise>();
    auto* graph = ctx.owner_graph(opr, inputs);
    return graph->insert_opr(
            std::make_unique<ForceInplaceElemwise>(inputs, opr.m_param, config));
}

MGB_REG_OPR_SHALLOW_COPY(ForceInplaceElemwise, ForceInplaceElemwise::shallow_copy);

cg::OperatorNodeBase* apply_inplace_add_on_var_node(
        const OpDef& def, const VarNodeArray& inputs) {
    auto dest = inputs[0], delta = inputs[1], alpha = inputs[2], beta = inputs[3];
    auto mode = ForceInplaceElemwise::Param::Mode::FUSE_MUL_ADD4;
    return ForceInplaceElemwise::make({alpha, dest, beta, delta}, {mode, 1})
            .node()
            ->owner_opr();
}

SmallVector<TensorPtr> apply_inplace_add_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    mgb_assert(
            inputs[0]->blob().use_count() == 2 && inputs[0]->blob()->storage().unique(),
            "This inplace modification may change the elements of other tensors. "
            "Please set MEGENGINE_INPLACE_UPDATE to 0 to ensure the program runs "
            "correctly.");
    auto dest = inputs[0], delta = inputs[1], alpha = inputs[2], beta = inputs[3];
    auto tensor_to_scalar = [](const TensorPtr& tensor) -> float {
        return *tensor->get_value().ptr<float>();
    };
    DnnOprCaller<megdnn::AddUpdate> caller{dest->comp_node()};
    caller.op->param() = {tensor_to_scalar(alpha), tensor_to_scalar(beta)};
    caller.op->exec(dest->dev_tensor().as_megdnn(), delta->dev_tensor().as_megdnn());
    return {std::make_shared<Tensor>(dest->blob(), dest->offset(), dest->layout())};
}

void execute_inplace(
        const OpDef& def, SmallVector<TensorPtr> inputs, SmallVector<TensorPtr> outputs,
        SmallVector<TensorPtr> workspace) {
    apply_inplace_add_on_physical_tensor(def, inputs);
}

std::tuple<SmallVector<MemoryDesc>, SmallVector<MemoryDesc>>
infer_inplace_output_mem_desc(
        const OpDef& def, const SmallVector<TensorPtr>& inputs_tensors,
        const SmallVector<MemoryDesc>& inputs_mems) {
    auto dest = inputs_tensors[0];
    SmallVector<MemoryDesc> outputs = {
            {dest->layout(), 0, dest->comp_node(),
             StorageIdentifier::make(&inputs_mems[0])}};
    return {outputs, {}};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_inplace_add_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(inputs.size() == 4, "invalid input number for inplace_add");
    CompNode cn;
    for (auto&& input : inputs) {
        if (!cn.valid()) {
            cn = input.comp_node;
        } else {
            mgb_assert(input.comp_node == cn, "inputs should be in same comp_node");
        }
    }
    auto dest = inputs[0], delta = inputs[1], alpha = inputs[2], beta = inputs[3];
    bool succeed = dest.layout.ndim != 0;
    if (succeed) {
        mgb_assert(
                delta.layout.ndim == 0 || dest.layout.eq_shape(delta.layout),
                "dest and delta must have same shape");
        mgb_assert(
                alpha.layout.ndim == 0 || alpha.layout.eq_shape({1}),
                "alpha should be scalar");
        mgb_assert(
                beta.layout.ndim == 0 || beta.layout.eq_shape({1}),
                "beta should be scalar");
    }
    mgb_assert(alpha.layout.dtype == dtype::Float32(), "alpha should be float32");
    mgb_assert(beta.layout.dtype == dtype::Float32(), "beta should be float32");
    // inplace op result's desc value is changed
    return {{{dest.layout, dest.comp_node}}, succeed};
}

OP_TRAIT_REG(Elemwise, Elemwise, opr::Elemwise)
        .make_from_op_node(make_from_op_node)
        .decide_dispatch_mode(decide_dispatch_mode)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_device_tensornd(apply_on_device_tensornd)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_mem_desc(infer_output_mem_desc)
        .execute(execute)
        .fallback();

OP_TRAIT_REG(InplaceAdd, InplaceAdd, opr::AddUpdate)
        .apply_on_var_node(apply_inplace_add_on_var_node)
        .apply_on_physical_tensor(apply_inplace_add_on_physical_tensor)
        .infer_output_attrs_fallible(infer_inplace_add_output_attrs_fallible)
        .infer_output_mem_desc(infer_inplace_output_mem_desc)
        .execute(execute_inplace)
        .fallback();
}  // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
