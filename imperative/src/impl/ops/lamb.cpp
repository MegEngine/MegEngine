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
namespace lamb {

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    return layout_checker;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    mgb_assert(input_descs.size() == 4, "IndexingOneHot expects 4inputs");
    auto comp_node = input_descs[0].comp_node;
    auto comp_node1 = input_descs[1].comp_node;
    auto comp_node2 = input_descs[2].comp_node;
    auto&& m_t_1 = input_descs[0].layout;
    auto&& v_t_1 = input_descs[1].layout;
    auto&& lamb_param = input_descs[2].layout;
    auto&& grad = input_descs[3].layout;
    MGB_MARK_USED_VAR(grad);
    auto&& new_param = lamb_param;
    auto&& m_t = m_t_1;
    auto&& v_t = v_t_1;
    return {{{m_t, comp_node}, {v_t, comp_node1}, {new_param, comp_node2}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<LAMBUpdate>();
    auto&& m_t_1 = inputs[0];
    auto&& v_t_1 = inputs[1];
    auto&& lamb_param = inputs[2];
    auto&& grad = inputs[3];

    TensorLayout m_t_1_layout{m_t_1->layout()};
    TensorLayout v_t_1_layout{v_t_1->layout()};
    TensorLayout lamb_param_layout{lamb_param->layout()};

    auto m_t = Tensor::make(m_t_1_layout, m_t_1->comp_node());
    auto v_t = Tensor::make(v_t_1_layout, v_t_1->comp_node());
    auto new_param = Tensor::make(lamb_param_layout, lamb_param->comp_node());

    DnnOprCaller<megdnn::LAMBUpdate> dnn_opr{lamb_param->comp_node(), op.param()};
    dnn_opr.exec_with_ws(m_t_1, v_t_1, lamb_param, grad, m_t, v_t, new_param);
    return {m_t, v_t, new_param};
}

OP_TRAIT_REG(LAMBUpdate, LAMBUpdate)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();

}  // namespace lamb
}  // namespace
}  // namespace imperative
}  // namespace mgb
