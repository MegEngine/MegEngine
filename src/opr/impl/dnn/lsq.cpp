#include "megbrain/opr/dnn/lsq.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSQForward);
MEGDNN_OPR_INIT4(LSQForward, "lsq_fwd");

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LSQForward) {
    SymbolVarArray grad = LSQBackward::make(
            out_grad[0], opr.input(0), opr.input(1), opr.input(2), opr.input(3),
            opr.param());

    if (wrt_idx == 0) {
        return grad[0].node();
    } else if (wrt_idx == 1) {
        return reduce_sum(grad[1], GetVarShape::make(opr.input(wrt_idx))).node();
    } else {
        return nullptr;
    }
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSQBackward);

LSQBackward::LSQBackward(
        VarNode* y_grad, VarNode* x, VarNode* scale, VarNode* zero_point,
        VarNode* grad_scale, const Param& param, const OperatorNodeConfig& config)
        : Super({x->owner_graph(),
                 config,
                 "lsq_bwd",
                 {y_grad, x, scale, zero_point, grad_scale}},
                1, true) {
    init_megdnn_opr(*this, param);
    add_input({y_grad, x, scale, zero_point, grad_scale});
}

SymbolVarArray LSQBackward::make(
        SymbolVar y_grad, SymbolVar x, SymbolVar scale, SymbolVar zero_point,
        SymbolVar grad_scale, const Param& param, const OperatorNodeConfig& config) {
    auto&& out = x.node()->owner_graph()
                         ->insert_opr(std::make_unique<LSQBackward>(
                                 y_grad.node(), x.node(), scale.node(),
                                 zero_point.node(), grad_scale.node(), param, config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = out[i];
    }
    return ret;
}

void LSQBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(1)));
    mgr.register_shape_infer(output(1), ShapeInferDesc::make_identity(input(1)));
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::LSQBackward>::val);
}

void LSQBackward::init_output_dtype() {
    output(0)->dtype(input(1)->dtype());
    output(1)->dtype(input(2)->dtype());
}
