#include "megbrain/graph/symbol_var.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/physical_tensor.h"
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
namespace padding {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Padding&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::Padding::make(inputs[0], op.param());
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto comp_node = inputs[0]->comp_node();
    auto&& op_def = def.cast_final_safe<Padding>();
    DnnOprCaller<megdnn::Padding> dnn_op(comp_node, op_def.param());
    auto dst = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_op.deduce_layout(inputs[0]->layout());
        }
    }();
    auto out = Tensor::make(dst, comp_node);
    dnn_op.exec(inputs[0], out);
    return {out};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Padding>();
    auto&& inp = inputs[0];

    if (inp.layout.ndim == 0) {
        return {{{TensorLayout{inp.layout.dtype}, inp.comp_node, {}}}, false};
    }

    DnnOprHelper<megdnn::Padding> dnn_op(op_def.param());
    auto oup_layout = dnn_op.deduce_layout(inp.layout);
    return {{{oup_layout, inp.comp_node}}, true};
}

OP_TRAIT_REG(Padding, Padding, opr::Padding)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .fallback();

}  // namespace padding
}  // namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
