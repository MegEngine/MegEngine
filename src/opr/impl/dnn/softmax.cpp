#include "megbrain/opr/dnn/softmax.h"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/utility.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

/* ==================== SoftmaxForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(SoftmaxForward);

SoftmaxForward::SoftmaxForward(
        VarNode* inp, const Param& param, const OperatorNodeConfig& config)
        : Super{inp->owner_graph(), config, "softmax", {inp}} {
    init_megdnn_opr(*this, param);

    add_input({inp});
    output(0)->dtype(inp->dtype());
}

SymbolVar SoftmaxForward::make(
        SymbolVar inp, const Param& param, const OperatorNodeConfig& config) {
    auto out = inp.node()
                       ->owner_graph()
                       ->insert_opr(std::make_unique<SoftmaxForward>(
                               inp.node(), param, config))
                       ->output();

    return out[0];
}

void SoftmaxForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    out_shape[0] = inp_shape[0];
}

size_t SoftmaxForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return megdnn_opr()->get_workspace_in_bytes(
            {input_shapes[0], input(0)->dtype(), input(0)->format()},
            {output_shapes[0], output(0)->dtype(), output(0)->format()});
}

void SoftmaxForward::scn_do_execute() {
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output().back()));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(SoftmaxForward) {
    SymbolVar grad = SoftmaxBackward::make(opr.output(0), out_grad[0], opr.param());

    return grad.node();
}
#endif

// /* ==================== SoftmaxBackward ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(SoftmaxBackward);

SoftmaxBackward::SoftmaxBackward(
        VarNode* src, VarNode* diff, const Param& param,
        const OperatorNodeConfig& config)
        : Super({src->owner_graph(), config, "Softmax_backward", {src, diff}}, 0,
                true) {
    init_megdnn_opr(*this, param);
    add_input({src, diff});
}

SymbolVar SoftmaxBackward::make(
        SymbolVar src, SymbolVar diff, const Param& param,
        const OperatorNodeConfig& config) {
    auto out = src.node()
                       ->owner_graph()
                       ->insert_opr(std::make_unique<SoftmaxBackward>(
                               src.node(), diff.node(), param, config))
                       ->output();
    return out[0];
}

void SoftmaxBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));
    this->init_output_static_infer_desc_workspace(false);
}

void SoftmaxBackward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

size_t SoftmaxBackward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return megdnn_opr()->get_workspace_in_bytes(
            {input_shapes[0], input(0)->dtype(), input(0)->format()},
            {input_shapes[1], input(1)->dtype(), input(1)->format()},
            {output_shapes[0], output(0)->dtype(), output(0)->format()});
}

void SoftmaxBackward::scn_do_execute() {
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
            output(0)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output().back()));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
