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

    DnnOprCaller<megdnn::Padding> dnn_op(comp_node);
    dnn_op.op->param() = op_def.param();

    TensorLayout dst = output_descs[0].layout;
    if (!validated) {
        megdnn::Padding::deduce_layout_impl(
                inputs[0]->dnn_tensor().layout, dst, op_def.param());
    }

    DeviceTensorND out =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, dst);

    dnn_op.op->exec(inputs[0]->dnn_tensor(), out.as_megdnn());

    return {Tensor::make(out)};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Padding>();
    size_t nr_inp = inputs.size();
    auto p = op_def.param();

    auto&& inp = inputs[0];
    auto& inp_cn = inp.comp_node;

    if (inp.layout.ndim == 0) {
        return {{{TensorLayout{inp.layout.dtype}, inp_cn, {}}}, false};
    }

    TensorLayout oup_layout;
    megdnn::Padding::deduce_layout_impl(inp.layout, oup_layout, p);
    return {{{oup_layout, inp_cn, {}}}, true};
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