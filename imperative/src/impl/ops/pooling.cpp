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
    if (!inputs[0].layout.ndim) {
        return {{{inputs[0].layout, inputs[0].comp_node}}, false};
    }
    DnnOprHelper<megdnn::Pooling> dnn_opr(op_def.param());
    auto oup_layout = dnn_opr.deduce_layout(inputs[0].layout);
    return {{{oup_layout, inputs[0].comp_node}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    mgb_assert(
            inputs.size() == 1, "num of inputs of pooling should be 1 but you give %zu",
            inputs.size());

    auto&& pooling = def.cast_final_safe<Pooling>();
    auto cn = inputs[0]->comp_node();
    DnnOprCaller<megdnn::Pooling> dnn_opr(cn, pooling.param(), pooling.policy());
    auto oup_layout = [&] {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_opr.deduce_layout(inputs[0]->layout());
        }
    }();

    auto&& src_layout = inputs[0]->layout();
    if (src_layout.ndim == 4 && src_layout.is_empty()) {
        if (pooling.param().format == megdnn::Pooling::Param::Format::NCHW) {
            mgb_assert(
                    src_layout.shape[2] != 0 && src_layout.shape[3] != 0,
                    "Pooling expect input to have non-zero size for non-batch "
                    "dimensions, but the input has layout %s",
                    src_layout.to_string().c_str());
        } else if (pooling.param().format == megdnn::Pooling::Param::Format::NHWC) {
            mgb_assert(
                    src_layout.shape[1] != 0 && src_layout.shape[2] != 0,
                    "Pooling expect input to have non-zero size for non-batch "
                    "dimensions, but the input has layout %s",
                    src_layout.to_string().c_str());
        } else {
            mgb_assert(
                    false,
                    "Pooling support empty input only when the input format is NHWC or "
                    "NCHW");
        }
        return {Tensor::make(oup_layout, cn)};
    }

    auto out = Tensor::make(oup_layout, cn);
    dnn_opr.exec_fastrun(inputs[0], out);
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
