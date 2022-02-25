#include "megbrain/opr/dnn/pooling.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/search_policy/algo_chooser.h"

#include "../search_policy/workspace_need_limit_getter.inl"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PoolingForward);

PoolingForward::PoolingForward(
        VarNode* i0, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{i0->owner_graph(), config, "pooling", {i0}}) {
    init_megdnn_opr(*this, param);
    add_input({i0});
    m_policy = policy;

    intl::MegDNNOprInitPostCtor<PoolingForward>::apply(*this);
}

SymbolVar PoolingForward::make(
        SymbolVar i0, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config) {
    intl::MegDNNOprInitInputsModifier<PoolingForward>::apply(param, {&i0});
    return i0.insert_single_output_opr<PoolingForward>(
            i0.node(), param, policy, config);
}

void PoolingForward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::Super::init_output_static_infer_desc();
    init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::PoolingForward>::val);
}

size_t PoolingForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return AlgoChooser<megdnn::PoolingForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(), input(0)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this, false);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(PoolingForward) {
    mgb_assert(wrt_idx == 0);
    SymbolVar grad = PoolingBackward::make(
            opr.input(0), opr.output(0), out_grad[0], opr.param(),
            opr.execution_policy());
    return grad.node();
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PoolingBackward);

PoolingBackward::PoolingBackward(
        VarNode* i0, VarNode* i1, VarNode* i2, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config)
        : Super(
                  OperatorNodeBaseCtorParam{
                          i0->owner_graph(), config, "pooling_bwd", {i0}},
                  0, true) {
    init_megdnn_opr(*this, param);
    add_input({i0, i1, i2});
    m_policy = policy;
    intl::MegDNNOprInitPostCtor<PoolingBackward>::apply(*this);
}

SymbolVar PoolingBackward::make(
        SymbolVar i0, SymbolVar i1, SymbolVar i2, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config) {
    intl::MegDNNOprInitInputsModifier<PoolingBackward>::apply(param, {&i0, &i1, &i2});
    return i0.insert_single_output_opr<PoolingBackward>(
            i0.node(), i1.node(), i2.node(), param, policy, config);
}

size_t PoolingBackward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return AlgoChooser<megdnn::PoolingBackward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(), input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {input_shapes[2], input(2)->dtype(), input(2)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this, false);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
