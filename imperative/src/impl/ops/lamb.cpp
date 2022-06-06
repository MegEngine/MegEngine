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
    TensorLayout m_t_1 = input_descs[0].layout, v_t_1 = input_descs[1].layout,
                 lamb_param = input_descs[2].layout, grad = input_descs[3].layout;

    TensorLayout new_param = lamb_param, m_t = m_t_1, v_t = v_t_1;
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

    DnnOprCaller<megdnn::LAMBUpdate> caller{lamb_param->comp_node()};
    size_t sz = caller.op->get_workspace_in_bytes(
            m_t_1->layout(), v_t_1->layout(), lamb_param->layout(), grad->layout(),
            m_t->layout(), v_t->layout(), new_param->layout());

    auto dnn_workspace = caller.create_workspace(sz);
    caller.op->param() = op.param();
    caller.op->exec(
            m_t_1->dev_tensor().as_megdnn(), v_t_1->dev_tensor().as_megdnn(),
            lamb_param->dev_tensor().as_megdnn(), grad->dev_tensor().as_megdnn(),
            m_t->dnn_tensor(), v_t->dnn_tensor(), new_param->dnn_tensor(),
            dnn_workspace);
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
