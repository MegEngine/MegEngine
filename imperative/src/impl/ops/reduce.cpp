#include "megbrain/graph/symbol_var.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megdnn/dtype.h"

#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {
namespace {
namespace reduce {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& reduce = static_cast<const Reduce&>(def);
    auto comp_node = inputs[0]->comp_node();
    OperatorNodeConfig config{reduce.make_name(), comp_node, inputs[0]->dtype()};

    if (inputs.size() > 1) {
        return opr::Reduce::make(inputs[0], reduce.param(), inputs[1], config);
    }

    using Param = megdnn::param::Reduce;
    auto param = reduce.param();
    if (param.axis < 0) {
        param.axis = inputs[0]->shape().ndim + param.axis;
    }

    SymbolVar target_shape = (cg::VarNode*)nullptr;
    if (param.axis == INT_MAX) {
        DTypeScalar vi{1};
        // auto graph = ComputingGraph::make();
        auto graph = inputs[0]->owner_graph();
        target_shape = opr::ImmutableTensor::make(*graph, vi, config);
    }
    auto res = opr::Reduce::make(inputs[0], param, target_shape, config);
    if (!reduce.keepdim && param.axis != INT_MAX) {
        using Desc = opr::AxisAddRemove::AxisDesc;
        std::vector<Desc> remove_param;
        remove_param.push_back(Desc::make_remove(param.axis));
        OperatorNodeConfig remove_config{
                def.make_name(), comp_node, inputs[0]->dtype()};
        return opr::AxisAddRemove::make(res, remove_param, remove_config);
    }
    return res;
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Reduce>();
    return Reduce::make(node->param(), true);
}

// TODO: using this for apply_on_physical_tensor
bool memory_forward_success(const OpDef& def, SmallVector<TensorPtr> inputs) {
    auto&& reduce = static_cast<const Reduce&>(def);
    if (reduce.mode != Reduce::Mode::SUM_SQR && inputs.size() == 2) {
        auto shape_tensor = inputs[1]->get_value();
        TensorShape shape;
        cg::copy_tensor_value_to_shape(shape, shape_tensor.proxy_to_default_cpu());
        if (shape.eq_shape(inputs[0]->shape())) {
            return true;
        }
    }
    return false;
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    if (memory_forward_success(def, inputs)) {
        return {Tensor::make(
                inputs[0]->blob(), inputs[0]->offset(), inputs[0]->layout())};
    }

    auto size = inputs.size();
    if (size > 1) {
        return proxy_graph_detail::apply_on_physical_tensor(
                def, inputs, output_descs, validated);
    }

    auto comp_node = inputs[0]->comp_node();
    using TensorND = megdnn::TensorND;
    auto&& op_def = def.cast_final_safe<Reduce>();
    SmallVector<TensorND> inp_tensornds;
    inp_tensornds.reserve(inputs.size());
    auto src = inputs[0]->layout();

    DnnOprCaller<megdnn::Reduce> dnn_op(comp_node);
    dnn_op.op->param() = op_def.param();
    auto axis = op_def.param().axis;
    auto keepdim = op_def.keepdim;

    if (axis < 0) {
        axis = inputs[0]->layout().ndim + axis;
    }

    dnn_op.op->param().axis = axis == INT_MAX ? 0 : axis;

    if (axis == INT_MAX) {
        src.shape[0] = src.total_nr_elems();
        src.ndim = 1;
        src.init_contiguous_stride();
    }
    TensorLayout layout{src.dtype};
    dnn_op.op->deduce_layout(src, layout);

    if (inputs[0]->layout().is_empty()) {
        inputs[0]->dev_tensor().reset(inputs[0]->dev_tensor().storage(), src);

        auto mode = op_def.param().mode;

        if (!keepdim && src.ndim > 1) {
            layout.remove_axis_inplace(axis);
            layout.init_contiguous_stride();
        }
        auto out = Tensor::make(layout, comp_node);

        std::string err_msg;
        switch (mode) {
            case Reduce::Mode::SUM:
                if (!out->empty()) {
                    dev_tensor_memset(out->dev_tensor(), 0);
                }
                break;
            case Reduce::Mode::PRODUCT:
                if (!out->empty()) {
                    DnnOprCaller<megdnn::Fill> fill_op(comp_node);
                    fill_op.op->param() = 1;
                    fill_op.op->exec(out->dnn_tensor(), {});
                }
                break;
            case Reduce::Mode::MEAN:
                err_msg = "mean";
                break;
            case Reduce::Mode::MIN:
                err_msg = "min";
                break;
            case Reduce::Mode::MAX:
                err_msg = "max";
                break;
            case Reduce::Mode::SUM_SQR:
                err_msg = "sum_sqr";
                break;
            default:
                mgb_throw(MegBrainError, "bad reduce mode");
        }
        if (!err_msg.empty()) {
            mgb_throw(
                    MegBrainError, "empty input is not allowed for reduce mode: %s",
                    err_msg.c_str());
        }
        return {out};
    }

    auto dnn_ten = inputs[0]->dnn_tensor();
    dnn_ten.layout = src;
    inp_tensornds.push_back(dnn_ten);

    auto wk_size = dnn_op.op->get_workspace_in_bytes(src, layout);
    auto dnn_wk = dnn_op.create_workspace(wk_size);
    TensorLayout ori_layout = layout;

    if (!keepdim && src.ndim > 1) {
        layout.remove_axis_inplace(axis);
        layout.init_contiguous_stride();
    }

    auto out = Tensor::make(layout, comp_node);
    auto dnn_out = out->dnn_tensor();
    dnn_out.layout = ori_layout;

    dnn_op.op->exec(inp_tensornds[0], dnn_out, dnn_wk);

    return {out};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Reduce>();
    auto axis = op_def.param().axis;
    auto keepdim = op_def.keepdim;

    size_t size = inputs.size();
    SmallVector<LogicalTensorDesc> dests(size);

    for (size_t i = 0; i < size; i++) {
        if (inputs[i].layout.ndim == 0) {
            return {{{TensorLayout(inputs[0].layout.dtype), inputs[0].comp_node}},
                    false};
        }
    }
    if (size > 1) {
        auto [output_descs, validated] =
                proxy_graph_detail::infer_output_attrs_fallible(def, inputs);
        if (!inputs[1].value.empty()) {
            cg::copy_tensor_value_to_shape(output_descs[0].layout, inputs[1].value);
            output_descs[0].layout.init_contiguous_stride();
        }
        return {output_descs, validated};
    }

    if (axis < 0) {
        axis = inputs[0].layout.ndim + axis;
    }

    if (axis == INT_MAX || inputs[0].layout.ndim == 1) {
        TensorLayout layout{inputs[0].layout.dtype};
        layout.shape[0] = 1;
        layout.ndim = 1;
        dests[0].layout = layout;
        dests[0].comp_node = inputs[0].comp_node;
    } else {
        for (size_t i = 0; i < size; ++i) {
            dests[i].comp_node = inputs[i].comp_node;
            dests[i].layout = inputs[i].layout;
            if (!keepdim && dests[i].layout.ndim > 1) {
                dests[i].layout.remove_axis_inplace(axis);
            } else {
                dests[i].layout.shape[axis] = 1;
            }
            dests[i].layout.init_contiguous_stride();
        }
    }

    return {dests, true};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(Reduce, Reduce, opr::Reduce)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace reduce
}  // namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
