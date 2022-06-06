#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/utility.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "../algo_chooser.h"
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb::imperative {

namespace {
namespace pooling {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& pool = static_cast<const Pooling&>(def);
    OperatorNodeConfig config{pool.make_name()};
    return opr::Pooling::make(inputs[0], pool.param(), pool.policy(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(
            inputs.size() == 1, "num of inputs of pooling should be 1 but you give %zu",
            inputs.size());

    auto&& op_def = def.cast_final_safe<Pooling>();
    auto&& inp = inputs[0];
    auto& inp_cn = inp.comp_node;

    if (inp.layout.ndim == 0) {
        return {{{TensorLayout{inp.layout.dtype}, inp_cn, {}}}, false};
    }

    TensorLayout oup_layout;
    megdnn::Pooling::deduce_layout_impl(inp.layout, op_def.param(), oup_layout);

    return {{{oup_layout, inp_cn, {}}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    mgb_assert(
            inputs.size() == 1, "num of inputs of pooling should be 1 but you give %zu",
            inputs.size());

    auto&& op_def = def.cast_final_safe<Pooling>();
    auto cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::Pooling> caller(cn);
    auto&& dnn_opr = caller.op;
    dnn_opr->param() = op_def.param();

    SmallVector<megdnn::TensorND> inp_tensornds(inputs.size());
    inp_tensornds[0] = inputs[0]->dnn_tensor();

    TensorLayout& oup_layout = output_descs[0].layout;
    if (!validated) {
        megdnn::Pooling::deduce_layout_impl(
                inp_tensornds[0].layout, op_def.param(), oup_layout);
    }

    size_t wk_size = setup_algo<megdnn::Pooling>(
            {inp_tensornds[0].layout, oup_layout}, dnn_opr.get(), 0, false, false, cn,
            op_def.policy(), false, &inp_tensornds);

    auto out = Tensor::make(oup_layout, cn);

    auto dnn_wk = caller.create_workspace(wk_size);

    caller.op->exec(inp_tensornds[0], out->dnn_tensor(), dnn_wk);
    return {out};
}

OP_TRAIT_REG(Pooling, Pooling)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace pooling
}  // namespace

}  // namespace mgb::imperative
