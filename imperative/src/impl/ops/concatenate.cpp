#include <climits>
#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/utils/stats.h"

namespace mgb::imperative {

namespace {

template <typename Opr>
CompNode get_device(const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Opr>();
    const char* op_name = op_def.make_name().c_str();
    CompNode oup_cn = op_def.comp_node;
    if (!oup_cn.valid()) {
        size_t nr_inp = inputs.size();
        mgb_assert(
                nr_inp > 0, "number of inputs of %s should be greater than 0", op_name);
        auto&& inp_cn = inputs[0].comp_node;
        for (size_t i = 1; i < nr_inp; ++i) {
            mgb_assert(
                    inp_cn == inputs[i].comp_node,
                    "input tensors of %s operator should have same device, but get "
                    "%s vs %s",
                    op_name, inp_cn.to_string().c_str(),
                    inputs[i].comp_node.to_string().c_str());
        }
        oup_cn = inp_cn;
    }
    return oup_cn;
}

bool is_all_inputs_valid(const SmallVector<LogicalTensorDesc>& inputs) {
    bool input_valid = true;
    size_t nr_inp = inputs.size();
    for (size_t i = 0; i < nr_inp; ++i) {
        if (inputs[i].layout.ndim == 0) {
            input_valid = false;
            break;
        }
    }
    return input_valid;
}

}  // namespace

namespace concatenate {

TensorLayout concat_layout_deduce(
        const SmallVector<const TensorLayout*> inputs, int axis) {
    // if we use megdnn::Concat::deduce_layout directly, we need construct
    // TensorLayoutArray, which will result in much memory copy
    auto shape_equal_but_specific_axis = [](const TensorShape& lhs,
                                            const TensorShape& rhs, int axis) -> bool {
        if (lhs.ndim != rhs.ndim) {
            return false;
        }
        for (size_t i = 0; i < lhs.ndim; ++i) {
            if (i == axis)
                continue;
            if (lhs.shape[i] != rhs.shape[i])
                return false;
        }
        return true;
    };

    TensorLayout oup_layout = *inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        mgb_assert(
                shape_equal_but_specific_axis(oup_layout, *inputs[i], axis),
                "Concat input shape mismatch: %s vs %s", inputs[0]->to_string().c_str(),
                inputs[i]->to_string().c_str());
        oup_layout.shape[axis] += inputs[i]->shape[axis];
    }
    oup_layout.init_contiguous_stride();
    return oup_layout;
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Concat&>(def);
    cg::OperatorNodeConfig config{op.comp_node};
    config.name(op.make_name());
    return opr::Concat::make(inputs, op.axis, config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Concat>();
    auto oup_cn = get_device<Concat>(def, inputs);

    if (!is_all_inputs_valid(inputs)) {
        // because dtypepromote_trans, so use inputs[0].dtype as oup_dtype here
        return {{{TensorLayout{inputs[0].layout.dtype}, oup_cn, {}}}, false};
    }

    SmallVector<const TensorLayout*> inputs_holder(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs_holder[i] = &inputs[i].layout;
    }
    int axis = op_def.axis >= 0 ? op_def.axis : op_def.axis + inputs[0].layout.ndim;
    TensorLayout oup_layout = concat_layout_deduce(inputs_holder, axis);
    return {{{oup_layout, oup_cn, {}}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<Concat>();
    int axis = op_def.axis >= 0 ? op_def.axis : op_def.axis + inputs[0]->layout().ndim;

    CompNode& oup_cn = output_descs[0].comp_node;
    if (op_def.comp_node.valid()) {
        mgb_assert(op_def.comp_node == oup_cn, "Concat compnode infer error");
    }

    // prepare inputs and output layout
    TensorLayout& oup_layout = output_descs[0].layout;
    if (!validated) {
        SmallVector<const TensorLayout*> inputs_holder(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs_holder[i] = &inputs[i]->layout();
        }
        oup_layout = concat_layout_deduce(inputs_holder, axis);
    }
    auto oup = Tensor::make(oup_layout, oup_cn);
    // because the dnn concat is very slow, we copy the slice code from
    // src/opr/impl/tensor_manip.cpp
    auto&& out = oup->dev_tensor();
    size_t end = 0;
    for (auto&& input : inputs) {
        auto&& in = input->dev_tensor();
        auto begin = end;
        end = begin + in.shape().shape[axis];
        if (!in.layout().is_empty()) {
            out.sub(Slice(begin, end).apply(out.layout(), axis))
                    .copy_from_fixlayout(in);
        }
    }
    return {oup};
}

OP_TRAIT_REG(Concat, Concat)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace concatenate

namespace stack {

TensorLayout stack_layout_deduce(
        const SmallVector<const TensorLayout*> inputs, int axis) {
    size_t nr_inp = inputs.size();
    auto&& inp_layout0 = *inputs[0];
    for (size_t i = 1; i < nr_inp; ++i) {
        mgb_assert(
                inp_layout0.eq_shape(*inputs[i]),
                "Stack input shape mismatch: %s vs %s", inp_layout0.to_string().c_str(),
                inputs[i]->to_string().c_str());
    }

    TensorLayout oup_layout{TensorShape{inp_layout0}, inp_layout0.dtype};
    oup_layout.add_axis_cont_inplace(axis);
    oup_layout.shape[axis] = nr_inp;
    oup_layout.init_contiguous_stride();
    return oup_layout;
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Stack&>(def);
    cg::OperatorNodeConfig config{op.comp_node};
    config.name(op.make_name());

    using Desc = opr::AxisAddRemove::AxisDesc;
    std::vector<Desc> param{Desc::make_add(op.axis)};
    VarNodeArray expanded_inputs;
    for (auto&& inp : inputs) {
        expanded_inputs.emplace_back(
                opr::AxisAddRemove::make(inp, param, cg::OperatorNodeConfig{}).node());
    }
    return opr::Concat::make(expanded_inputs, op.axis, config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Stack>();
    auto oup_cn = get_device<Stack>(def, inputs);

    if (!is_all_inputs_valid(inputs)) {
        // because dtypepromote_trans, so use inputs[0].dtype as oup_dtype here
        return {{{TensorLayout{inputs[0].layout.dtype}, oup_cn, {}}}, false};
    }

    SmallVector<const TensorLayout*> inputs_holder(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs_holder[i] = &inputs[i].layout;
    }
    int axis = op_def.axis >= 0 ? op_def.axis : op_def.axis + inputs[0].layout.ndim + 1;
    TensorLayout oup_layout = stack_layout_deduce(inputs_holder, axis);
    return {{{oup_layout, oup_cn, {}}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<Stack>();
    size_t nr_inp = inputs.size();
    TensorLayout inp_layout = inputs[0]->layout();
    int axis =
            op_def.axis >= 0 ? op_def.axis : op_def.axis + inputs[0]->layout().ndim + 1;

    CompNode& oup_cn = output_descs[0].comp_node;
    if (op_def.comp_node.valid()) {
        mgb_assert(op_def.comp_node == oup_cn, "Stack compnode infer error");
    }

    // prepare inputs and output layout
    TensorLayout& oup_layout = output_descs[0].layout;
    if (!validated) {
        SmallVector<const TensorLayout*> inputs_holder(inputs.size());
        for (size_t i = 0; i < nr_inp; ++i) {
            inputs_holder[i] = &inputs[i]->layout();
        }
        oup_layout = stack_layout_deduce(inputs_holder, axis);
    }
    inp_layout.add_axis_cont_inplace(axis);
    SmallVector<TensorPtr> expanded;
    for (size_t i = 0; i < nr_inp; ++i) {
        expanded.push_back(
                Tensor::make(inputs[i]->blob(), inputs[i]->offset(), inp_layout));
    }
    auto oup = Tensor::make(oup_layout, oup_cn);
    // because the dnn concat is very slow, we copy the slice code from
    // src/opr/impl/tensor_manip.cpp
    auto&& out = oup->dev_tensor();
    size_t end = 0;
    for (auto&& input : expanded) {
        auto&& in = input->dev_tensor();
        auto begin = end;
        end = begin + in.shape().shape[axis];
        if (!in.layout().is_empty()) {
            out.sub(Slice(begin, end).apply(out.layout(), axis))
                    .copy_from_fixlayout(in);
        }
    }
    return {oup};
}

OP_TRAIT_REG(Stack, Stack)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace stack
}  // namespace mgb::imperative
