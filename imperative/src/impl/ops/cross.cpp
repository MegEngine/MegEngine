#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/blas.h"
#include "megdnn/oprs.h"

#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {
namespace cross {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Cross>();
    mgb_assert(inputs.size() == 2);
    cg::OperatorNodeConfig config{op.make_name()};
    return opr::Cross::make(inputs[0], inputs[1], op.param(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(inputs.size() == 2, "Cross expects two inputs");
    auto&& op_def = def.cast_final_safe<Cross>();
    auto comp_node = inputs[0].comp_node;

    if (!inputs[0].layout.ndim) {
        return {{{inputs[0].layout, comp_node}}, false};
    }
    if (!inputs[1].layout.ndim) {
        return {{{inputs[1].layout, comp_node}}, false};
    }

    DnnOprHelper<megdnn::Cross> dnn_op(op_def.param());
    auto oup_layout = dnn_op.deduce_layout(inputs[0].layout, inputs[1].layout);
    return {{{oup_layout, comp_node}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto comp_node = inputs[0]->comp_node();
    auto&& op_def = def.cast_final_safe<Cross>();
    DnnOprCaller<megdnn::Cross> dnn_op(comp_node, op_def.param());
    auto dst = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_op.deduce_layout(inputs[0]->layout(), inputs[1]->layout());
        }
    }();
    auto out = Tensor::make(dst, comp_node);
    if (!inputs[0]->layout().is_empty() && !inputs[1]->layout().is_empty()) {
        dnn_op.exec_with_ws(inputs[0], inputs[1], out);
    }
    return {out};
}

OP_TRAIT_REG(Cross, Cross)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace cross
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
