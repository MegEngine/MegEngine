#include "megbrain/opr/dnn/group_norm.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb::imperative {
namespace group_norm {

cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const GroupNorm&>(def);
    size_t nr_inp = inputs.size();
    auto p = op.param();
    mgb_assert((nr_inp == 3 && p.affine) || (nr_inp == 1 && !p.affine));
    OperatorNodeConfig config{op.make_name()};
    if (nr_inp == 3) {
        return opr::GroupNorm::make(
                       inputs[0], inputs[1], inputs[2], op.param(), config)[0]
                .node()
                ->owner_opr();
    } else {
        return opr::GroupNorm::make(inputs[0], op.param(), config)[0]
                .node()
                ->owner_opr();
    }
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& group_norm = def.cast_final_safe<GroupNorm>();
    size_t nr_inp = inputs.size();
    auto affine = group_norm.affine;
    mgb_assert(
            (nr_inp == 3 && affine) || (nr_inp == 1 && !affine),
            "num of inputs of pooling should be 1 or 3 but you give %zu",
            inputs.size());

    auto&& inp = inputs[0];
    auto& inp_cn = inp.comp_node;

    if (inp.layout.ndim == 0) {
        return {{{TensorLayout{inp.layout.dtype}, inp_cn, {}},
                 {TensorLayout{dtype::Float32()}, inp_cn, {}},
                 {TensorLayout{dtype::Float32()}, inp_cn, {}}},
                false};
    }

    DnnOprHelper<megdnn::GroupNorm> dnn_opr(group_norm.param());
    auto&& [oup_layout, mean_layout, rstd_layout] =
            dnn_opr.deduce_layouts<3>(inp.layout, TensorLayout{}, TensorLayout{});
    return {{{oup_layout, inp_cn, {}},
             {mean_layout, inp_cn, {}},
             {rstd_layout, inp_cn, {}}},
            true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<GroupNorm>();
    size_t nr_inp = inputs.size();
    auto p = op_def.param();

    mgb_assert(
            (nr_inp == 3 && p.affine) || (nr_inp == 1 && !p.affine),
            "num of inputs of groupnorm should be 1 or 3 but you give %zu",
            inputs.size());

    auto cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::GroupNorm> caller(cn, op_def.param());

    auto&& [oup_layout, mean_layout, rstd_layout] = caller.deduce_layouts<3>(
            inputs[0]->layout(), TensorLayout{}, TensorLayout{});

    auto out = Tensor::make(oup_layout, cn);
    auto mean = Tensor::make(mean_layout, cn);
    auto rstd = Tensor::make(rstd_layout, cn);

    if (p.affine) {
        caller.exec_with_ws(inputs[0], inputs[1], inputs[2], out, mean, rstd);
    } else {
        megdnn::TensorND empty_dnn;
        caller.exec_with_ws(inputs[0], empty_dnn, empty_dnn, out, mean, rstd);
    }
    return {out, mean, rstd};
}

OP_TRAIT_REG(GroupNorm, GroupNorm)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace group_norm
}  // namespace mgb::imperative
