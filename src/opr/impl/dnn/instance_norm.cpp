#include "megbrain/opr/dnn/instance_norm.h"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/utility.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

/* ==================== InstanceNormForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(InstanceNormForward);

InstanceNormForward::InstanceNormForward(
        VarNode* data, VarNode* weight, VarNode* bias, const Param& param,
        const OperatorNodeConfig& config)
        : Super{data->owner_graph(), config, "instance_norm", {data, weight, bias}} {
    init_megdnn_opr(*this, param);

    add_input({data, weight, bias});
    output(0)->dtype(data->dtype());
    output(1)->dtype(dtype::Float32());
    output(2)->dtype(dtype::Float32());
}

InstanceNormForward::InstanceNormForward(
        VarNode* data, const Param& param, const OperatorNodeConfig& config)
        : Super{data->owner_graph(), config, "instance_norm", {data}} {
    init_megdnn_opr(*this, param);

    add_input({data});
    output(0)->dtype(data->dtype());
    output(1)->dtype(dtype::Float32());
    output(2)->dtype(dtype::Float32());
}

SymbolVarArray InstanceNormForward::make(
        SymbolVar data, SymbolVar weight, SymbolVar bias, const Param& param,
        const OperatorNodeConfig& config) {
    auto outs = data.node()
                        ->owner_graph()
                        ->insert_opr(std::make_unique<InstanceNormForward>(
                                data.node(), weight.node(), bias.node(), param, config))
                        ->output();
    SymbolVarArray ret;
    for (auto&& out : outs) {
        ret.emplace_back(out);
    }
    return ret;
}

SymbolVarArray InstanceNormForward::make(
        SymbolVar data, const Param& param, const OperatorNodeConfig& config) {
    auto outs = data.node()
                        ->owner_graph()
                        ->insert_opr(std::make_unique<InstanceNormForward>(
                                data.node(), param, config))
                        ->output();
    SymbolVarArray ret;
    for (auto&& out : outs) {
        ret.emplace_back(out);
    }
    return ret;
}

void InstanceNormForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    out_shape[0] = inp_shape[0];
    size_t N = inp_shape[0].shape[0];
    size_t C = inp_shape[0].shape[1];
    TensorShape unnormalized_shape{N, C};
    out_shape[1] = unnormalized_shape;
    out_shape[2] = unnormalized_shape;
}

size_t InstanceNormForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
#define in(x) \
    { input_shapes[x], input(x)->dtype() }
#define out(x) \
    { output_shapes[x], output(x)->dtype() }
    auto&& param = megdnn_opr()->param();
    if (param.affine) {
        return megdnn_opr()->get_workspace_in_bytes(
                in(0), in(1), in(2), out(0), out(1), out(2));
    } else {
        TensorLayout temp_weight = TensorLayout(dtype::Float32());
        TensorLayout temp_bias = TensorLayout(dtype::Float32());
        return megdnn_opr()->get_workspace_in_bytes(
                in(0), temp_weight, temp_bias, out(0), out(1), out(2));
    }
#undef in
#undef out
}

void InstanceNormForward::scn_do_execute() {
    auto p = param();
    mgb_assert(p.format == Param::Format::NCHW, "only support inputs in shape NCHW.");
    size_t C = input(0)->dev_tensor().shape()[1];
    auto opr = const_cast<megdnn::GroupNormForward*>(megdnn_opr());
    opr->param().group = C;
    mgb_assert(C != 0, "error param!");
    if (param().affine) {
        megdnn_opr()->exec(
                input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
                input(2)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
                output(1)->dev_tensor().as_megdnn(),
                output(2)->dev_tensor().as_megdnn(),
                intl::get_megdnn_workspace_from_var(output().back()));
    } else {
        megdnn_opr()->exec(
                input(0)->dev_tensor().as_megdnn(), {}, {},
                output(0)->dev_tensor().as_megdnn(),
                output(1)->dev_tensor().as_megdnn(),
                output(2)->dev_tensor().as_megdnn(),
                intl::get_megdnn_workspace_from_var(output().back()));
    }
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(InstanceNormForward) {
    auto p = opr.param();
    SymbolVarArray grad;
    VarNodeArray ret;
    if (p.affine) {
        mgb_assert(wrt_idx < 3, "wrt_idx %zu is out of range", wrt_idx);
        grad = InstanceNormBackward::make(
                out_grad[0], opr.input(0), opr.input(1), opr.output(1), opr.output(2),
                opr.param());
    } else {
        mgb_assert(wrt_idx < 1, "wrt_idx %zu is out of range", wrt_idx);
        grad = InstanceNormBackward::make(
                out_grad[0], opr.input(0), opr.output(1), opr.output(2), opr.param());
    }

    uint32_t nr_ret = p.affine ? 3 : 1;
    for (uint32_t i = 0; i < nr_ret; ++i) {
        ret.push_back(grad[i].node());
    }
    return ret;
}
#endif

/* ==================== InstanceNormBackward ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(InstanceNormBackward);

InstanceNormBackward::InstanceNormBackward(
        VarNode* diff, VarNode* data, VarNode* weight, VarNode* mean, VarNode* rstd,
        const Param& param, const OperatorNodeConfig& config)
        : Super({diff->owner_graph(),
                 config,
                 "instance_norm_backward",
                 {diff, data, weight, mean, rstd}},
                0, true) {
    init_megdnn_opr(*this, param);
    add_input({diff, data, weight, mean, rstd});
}

InstanceNormBackward::InstanceNormBackward(
        VarNode* diff, VarNode* data, VarNode* mean, VarNode* rstd, const Param& param,
        const OperatorNodeConfig& config)
        : Super({diff->owner_graph(),
                 config,
                 "instance_norm_backward",
                 {diff, data, mean, rstd}},
                0, true) {
    init_megdnn_opr(*this, param);
    add_input({diff, data, mean, rstd});
    auto mark_empty_var = [&](VarNode* var) {
        var->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                .add_flag(VarNode::Flag::VOLATILE_CONTENT);
    };
    mark_empty_var(output(1));
    mark_empty_var(output(2));
}

SymbolVarArray InstanceNormBackward::make(
        SymbolVar diff, SymbolVar data, SymbolVar weight, SymbolVar mean,
        SymbolVar rstd, const Param& param, const OperatorNodeConfig& config) {
    auto outs = diff.node()
                        ->owner_graph()
                        ->insert_opr(std::make_unique<InstanceNormBackward>(
                                diff.node(), data.node(), weight.node(), mean.node(),
                                rstd.node(), param, config))
                        ->output();
    SymbolVarArray ret;
    for (auto&& out : outs) {
        ret.emplace_back(out);
    }
    return ret;
}

SymbolVarArray InstanceNormBackward::make(
        SymbolVar diff, SymbolVar data, SymbolVar mean, SymbolVar rstd,
        const Param& param, const OperatorNodeConfig& config) {
    auto outs = diff.node()
                        ->owner_graph()
                        ->insert_opr(std::make_unique<InstanceNormBackward>(
                                diff.node(), data.node(), mean.node(), rstd.node(),
                                param, config))
                        ->output();
    SymbolVarArray ret;
    for (auto&& out : outs) {
        ret.emplace_back(out);
    }
    return ret;
}

void InstanceNormBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(1)));
    if (param().affine) {
        mgr.register_shape_infer(output(1), ShapeInferDesc::make_identity(input(2)));
        mgr.register_shape_infer(output(2), ShapeInferDesc::make_identity(input(2)));
    } else {
        TensorShape empty;
        empty.ndim = 0;
        mgr.register_shape_infer(output(1), ShapeInferDesc::make_const(empty));
        mgr.register_shape_infer(output(2), ShapeInferDesc::make_const(empty));
    }
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::GroupNormBackward>::val);
}

void InstanceNormBackward::init_output_dtype() {
    output(0)->dtype(input(1)->dtype());
    output(1)->dtype(input(2)->dtype());
    output(2)->dtype(input(2)->dtype());
}

size_t InstanceNormBackward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
#define in(x) \
    { input_shapes[x], input(x)->dtype() }
#define out(x) \
    { output_shapes[x], output(x)->dtype() }
    auto&& param = megdnn_opr()->param();
    if (param.affine) {
        return megdnn_opr()->get_workspace_in_bytes(
                in(0), in(1), in(2), in(3), in(4), out(0), out(1), out(2));
    } else {
        TensorLayout temp_weight = TensorLayout(dtype::Float32());
        TensorLayout temp_dweight = TensorLayout(dtype::Float32());
        TensorLayout temp_dbias = TensorLayout(dtype::Float32());
        return megdnn_opr()->get_workspace_in_bytes(
                in(0), in(1), temp_weight, in(2), in(3), out(0), temp_dweight,
                temp_dbias);
    }
#undef in
#undef out
}

void InstanceNormBackward::scn_do_execute() {
    auto p = param();
    mgb_assert(p.format == Param::Format::NCHW, "only support inputs in shape NCHW.");
    size_t C = input(0)->dev_tensor().shape()[1];
    auto opr = const_cast<megdnn::GroupNormBackward*>(megdnn_opr());
    opr->param().group = C;
    mgb_assert(C != 0, "error param!");

    if (p.affine) {
        megdnn_opr()->exec(
                input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
                input(2)->dev_tensor().as_megdnn(), input(3)->dev_tensor().as_megdnn(),
                input(4)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
                output(1)->dev_tensor().as_megdnn(),
                output(2)->dev_tensor().as_megdnn(),
                intl::get_megdnn_workspace_from_var(output(3)));
    } else {
        megdnn_opr()->exec(
                input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
                {}, input(2)->dev_tensor().as_megdnn(),
                input(3)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
                {}, {}, intl::get_megdnn_workspace_from_var(output(3)));
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
