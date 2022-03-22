#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/stats.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/utility.h"

#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace {
namespace dot {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Dot>();
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::Dot::make(inputs[0], inputs[1], config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto comp_node = inputs[0]->comp_node();
    using TensorND = megdnn::TensorND;
    SmallVector<TensorND> inp_tensornds;
    inp_tensornds.reserve(inputs.size());
    DnnOprCaller<megdnn::Dot> dnn_opr(comp_node);
    for (unsigned i = 0; i < inputs.size(); ++i) {
        auto dnn_ten = inputs[i]->dnn_tensor();
        inp_tensornds.push_back(dnn_ten);
    }
    TensorLayout oup_layout{inputs[0]->dtype()};
    auto inp1_tensor = inputs[0]->dnn_tensor();
    auto inp2_tensor = inputs[1]->dnn_tensor();
    dnn_opr.op->deduce_layout(inp1_tensor.layout, inp2_tensor.layout, oup_layout);

    if (inputs[0]->layout().is_empty() || inputs[1]->layout().is_empty()) {
        DnnOprCaller<megdnn::Fill> fill_opr(comp_node);
        DeviceTensorND out =
                BlobManager::inst()->alloc_workspace_with_defrag(comp_node, oup_layout);
        fill_opr.op->param() = 0;
        fill_opr.op->exec(out.as_megdnn(), {});
        return {Tensor::make(out)};
    }

    auto sz = dnn_opr.op->get_workspace_in_bytes(
            inp_tensornds[0].layout, inp_tensornds[1].layout, output_descs[0].layout);

    DeviceTensorND out_devtensor =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, oup_layout);

    TensorLayout w_layout({sz}, dtype::Byte());
    auto dnn_wk = dnn_opr.create_workspace(w_layout);

    dnn_opr.op->exec(
            inp_tensornds[0], inp_tensornds[1], out_devtensor.as_megdnn(), dnn_wk);

    return {Tensor::make(out_devtensor)};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(
            inputs.size() == 2, "Dot expects 2 inputs; got %lu actually",
            inputs.size());
    SmallVector<LogicalTensorDesc> dests(1);
    dests[0].layout = TensorLayout(TensorShape{1}, inputs[0].layout.dtype);
    dests[0].comp_node = inputs[0].comp_node;
    bool validated = inputs[0].layout.ndim != 0 && inputs[1].layout.ndim != 0;
    return {dests, validated};
}

OP_TRAIT_REG(Dot, Dot, mgb::opr::Dot)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace dot
}  // anonymous namespace
}  // namespace imperative
}  // namespace mgb