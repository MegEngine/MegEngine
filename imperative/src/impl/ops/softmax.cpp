#include "megbrain/opr/dnn/softmax.h"
#include "megbrain/imperative/ops/autogen.h"

#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {
namespace {
namespace softmax {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& softmax = static_cast<const Softmax&>(def);
    OperatorNodeConfig config{softmax.make_name()};
    return opr::Softmax::make(inputs[0], softmax.param(), config);
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Softmax>();
    return Softmax::make(node->param());
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef&, const SmallVector<LogicalTensorDesc>& inputs) {
    SmallVector<LogicalTensorDesc> out_shapes(1);
    auto&& i0 = inputs[0];
    out_shapes[0] = {i0.layout, i0.comp_node};
    return {out_shapes, true};
}

OP_TRAIT_REG(Softmax, Softmax, opr::Softmax)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .fallback();

}  // namespace softmax
}  // namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}