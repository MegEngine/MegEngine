#include "megbrain/opr/dnn/layer_norm.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb::imperative {
namespace layer_norm {

cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const LayerNorm&>(def);
    size_t nr_inp = inputs.size();
    auto p = op.param();
    mgb_assert((nr_inp == 3 && p.affine) || (nr_inp == 1 && !p.affine));
    OperatorNodeConfig config{op.make_name()};
    if (nr_inp == 3) {
        return opr::LayerNorm::make(
                       inputs[0], inputs[1], inputs[2], op.param(), config)[0]
                .node()
                ->owner_opr();
    } else {
        return opr::LayerNorm::make(inputs[0], op.param(), config)[0]
                .node()
                ->owner_opr();
    }
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<LayerNorm>();
    size_t nr_inp = inputs.size();
    auto p = op_def.param();
    mgb_assert(
            (nr_inp == 3 && p.affine) || (nr_inp == 1 && !p.affine),
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

    TensorLayout oup_layout, mean_layout, rstd_layout;
    megdnn::LayerNorm::deduce_layout_fwd_impl(
            inp.layout, p, oup_layout, mean_layout, rstd_layout);
    return {{{oup_layout, inp_cn, {}},
             {mean_layout, inp_cn, {}},
             {rstd_layout, inp_cn, {}}},
            true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<LayerNorm>();
    size_t nr_inp = inputs.size();
    auto p = op_def.param();

    mgb_assert(
            (nr_inp == 3 && p.affine) || (nr_inp == 1 && !p.affine),
            "num of inputs of pooling should be 1 or 3 but you give %zu",
            inputs.size());

    auto cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::LayerNorm> caller(cn);
    auto&& dnn_opr = caller.op;
    dnn_opr->param() = p;

    TensorLayout oup_layout, mean_layout, rstd_layout;
    megdnn::LayerNorm::deduce_layout_fwd_impl(
            inputs[0]->dnn_tensor().layout, p, oup_layout, mean_layout, rstd_layout);

    DeviceTensorND out_devtensor =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, oup_layout);
    DeviceTensorND mean_devtensor =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, mean_layout);
    DeviceTensorND rstd_devtensor =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, rstd_layout);

    megdnn::Workspace dnn_wk;
    auto wk_size = caller.op->get_workspace_in_bytes(
            inputs[0]->dnn_tensor().layout,
            p.affine ? inputs[1]->dnn_tensor().layout : TensorLayout(),
            p.affine ? inputs[2]->dnn_tensor().layout : TensorLayout(), oup_layout,
            mean_layout, rstd_layout);
    if (wk_size != 0) {
        TensorLayout w_layout({wk_size}, dtype::Byte());
        dnn_wk = caller.create_workspace(w_layout);
    }

    dnn_opr->exec(
            inputs[0]->dnn_tensor(),
            p.affine ? inputs[1]->dnn_tensor() : megdnn::TensorND(),
            p.affine ? inputs[2]->dnn_tensor() : megdnn::TensorND(),
            out_devtensor.as_megdnn(), mean_devtensor.as_megdnn(),
            rstd_devtensor.as_megdnn(), dnn_wk);
    return {Tensor::make(out_devtensor), Tensor::make(mean_devtensor),
            Tensor::make(rstd_devtensor)};
}

OP_TRAIT_REG(LayerNorm, LayerNorm)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace layer_norm
}  // namespace mgb::imperative