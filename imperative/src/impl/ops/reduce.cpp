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
    auto name = reduce.make_name();

    auto param = reduce.param();
    auto axis = param.axis;
    auto keepdim = reduce.keepdim;

    if (inputs.size() == 2) {
        return opr::Reduce::make(inputs[0], param, inputs[1], {name});
    }
    mgb_assert(inputs.size() == 1);

    if (axis == INT_MAX) {
        // keepdim could be ignored when ndim == 1
        auto graph = inputs[0]->owner_graph();
        auto scalar_shape =
                opr::ImmutableTensor::make(*graph, DTypeScalar(1), {name, comp_node});
        return opr::Reduce::make(inputs[0], param, scalar_shape, {name});
    }
    // mgb::opr::Reduce supports negative axis
    auto res = opr::Reduce::make(inputs[0], param, {}, {name});
    if (!keepdim) {
        using Desc = opr::AxisAddRemove::AxisDesc;
        std::vector<Desc> remove_axis_param;
        remove_axis_param.push_back(Desc::make_remove(axis));
        res = opr::AxisAddRemove::make(res, remove_axis_param, {name});
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
    // memory forward
    if (memory_forward_success(def, inputs)) {
        // maybe returns inputs[0] directly
        return {Tensor::make(
                inputs[0]->blob(), inputs[0]->offset(), inputs[0]->layout())};
    }

    if (inputs.size() == 2) {
        // reduce to target shape, fallback to proxy_graph
        return proxy_graph_detail::apply_on_physical_tensor(
                def, inputs, output_descs, validated);
    }
    mgb_assert(inputs.size() == 1);

    auto comp_node = inputs[0]->comp_node();
    auto&& op_def = def.cast_final_safe<Reduce>();
    DnnOprCaller<megdnn::Reduce> dnn_op(comp_node, op_def.param());
    auto&& mode = dnn_op.param().mode;
    auto& axis = dnn_op.param().axis;
    auto keepdim = op_def.keepdim;

    DnnTensorND dnn_input = [&] {
        if (axis == INT_MAX) {  // reduce to scalar
            axis = 0;
            // flatten input
            return inputs[0]->dnn_tensor({inputs[0]->shape().total_nr_elems()});
        } else {
            if (axis < 0) {
                axis = inputs[0]->layout().ndim + axis;
            }
            mgb_assert(axis >= 0 && axis < inputs[0]->layout().ndim);
            return inputs[0]->dnn_tensor();
        }
    }();
    auto output_layout = dnn_op.deduce_layout(dnn_input.layout);
    auto resolve_keepdim = [&] {
        if (!keepdim) {
            if (output_layout.ndim > 1) {
                mgb_assert(output_layout.shape[axis] == 1);
                output_layout.remove_axis_inplace(axis);
            }
        }
    };

    TensorPtr output;
    if (output_layout.is_empty()) {
        // output empty, no computation
        resolve_keepdim();
        output = Tensor::make(output_layout, comp_node);
    } else if (dnn_input.layout.is_empty()) {
        // input empty but output not, do fill
        resolve_keepdim();
        output = Tensor::make(output_layout, comp_node);
        auto on_bad_empty_reduce = [](const char* name) {
            mgb_throw(
                    MegBrainError, "empty input is not allowed for reduce mode: %s",
                    name);
        };
        switch (mode) {
            case Reduce::Mode::SUM:
                // fill 0
                dev_tensor_memset(output->dev_tensor(), 0);
                break;
            case Reduce::Mode::PRODUCT: {
                // fill 1
                DnnOprCaller<megdnn::Fill> fill_op(comp_node, {1});
                fill_op.exec_with_ws(output);
                break;
            }
            case Reduce::Mode::MEAN:
                on_bad_empty_reduce("mean");
                break;
            case Reduce::Mode::MIN:
                on_bad_empty_reduce("min");
                break;
            case Reduce::Mode::MAX:
                on_bad_empty_reduce("max");
                break;
            case Reduce::Mode::SUM_SQR:
                on_bad_empty_reduce("sum_sqr");
                break;
            default:
                mgb_throw(MegBrainError, "bad reduce mode");
        }
    } else {
        // common reduction
        if (keepdim) {
            output = Tensor::make(output_layout, comp_node);
            dnn_op.exec_with_ws(dnn_input, output);
        } else {
            // used by megdnn::exec
            auto output_layout_keepdim = output_layout;
            resolve_keepdim();
            output = Tensor::make(output_layout, comp_node);
            dnn_op.exec_with_ws(dnn_input, output->dnn_tensor(output_layout_keepdim));
        }
    }
    return {output};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Reduce>();
    auto axis = op_def.param().axis;
    auto keepdim = op_def.keepdim;

    mgb_assert(inputs.size() > 0);
    auto&& comp_node = inputs[0].comp_node;
    auto&& input_layout = inputs[0].layout;

    if (inputs.size() == 2) {
        // fallback to proxy_graph, matters on backward
        auto [output_descs, validated] =
                proxy_graph_detail::infer_output_attrs_fallible(def, inputs);
        if (!inputs[1].value.empty()) {
            cg::copy_tensor_value_to_shape(output_descs[0].layout, inputs[1].value);
            output_descs[0].layout.init_contiguous_stride();
        }
        return {output_descs, validated};
    }

    mgb_assert(inputs.size() == 1);

    if (axis == INT_MAX) {
        // reduce to scalar
        // ignore keepdim because ndim is 1
        auto&& dtype = input_layout.dtype;
        auto&& format = input_layout.format;
        auto output_layout = TensorLayout{{1}, dtype, format};
        return {{{output_layout, comp_node}}, true};
    }

    if (input_layout.ndim == 0) {
        // shape incomplete
        return {{{TensorLayout(input_layout.dtype, input_layout.format), comp_node}},
                false};
    }

    if (axis < 0) {
        axis = input_layout.ndim + axis;
    }
    mgb_assert(axis >= 0 && axis < input_layout.ndim);

    TensorLayout output_layout = input_layout;
    bool remove_axis = (!keepdim) && input_layout.ndim > 1;
    if (remove_axis) {
        output_layout.remove_axis_inplace(axis);
    } else {
        output_layout.shape[axis] = 1;
    }
    output_layout.init_contiguous_stride();
    return {{{output_layout, comp_node}}, true};
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
