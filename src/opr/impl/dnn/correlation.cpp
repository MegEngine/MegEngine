#include "megbrain/opr/dnn/correlation.h"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/utility.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

/* ==================== CorrelationForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CorrelationForward);
CorrelationForward::CorrelationForward(
        VarNode* data1, VarNode* data2, const Param& param,
        const OperatorNodeConfig& config)
        : Super{data1->owner_graph(), config, "correlation", {data1, data2}} {
    init_megdnn_opr(*this, param);
    mgb_assert(data1->dtype() == data2->dtype());
    mgb_assert(data1->dtype().category() == DTypeCategory::FLOAT);

    add_input({data1, data2});
    output(0)->dtype(data1->dtype());
}

SymbolVar CorrelationForward::make(
        SymbolVar data1, SymbolVar data2, const Param& param,
        const OperatorNodeConfig& config) {
    return data1.insert_single_output_opr<CorrelationForward>(
            data1.node(), data2.node(), param, config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(CorrelationForward) {
    if (wrt_idx == 0) {
        // wrt src
        SymbolVar grad = CorrelationBackwardData1::make(
                out_grad[0], opr.input(0), opr.input(1), opr.param(), opr.config());
        return grad.node();
    } else {
        mgb_assert(wrt_idx == 1);
        SymbolVar grad = CorrelationBackwardData2::make(
                out_grad[0], opr.input(0), opr.input(1), opr.param(), opr.config());
        return grad.node();
    }
}
#endif

/* ==================== CorrelationBackwardData1 ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CorrelationBackwardData1);
MEGDNN_OPR_INIT3(CorrelationBackwardData1, "correlation_backward_data1", 1, true);

void CorrelationBackwardData1::scn_do_execute() {
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
            input(2)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output(1)));
}

size_t CorrelationBackwardData1::get_workspace_size_bytes(
        const TensorShapeArray& inp_shapes, const TensorShapeArray& out_shapes) const {
    TensorLayout diff{inp_shapes[0], input(0)->dtype(), input(0)->format()},
            data1{inp_shapes[1], input(1)->dtype(), input(1)->format()},
            data2{inp_shapes[2], input(2)->dtype(), input(2)->format()},
            grad1{out_shapes[0], output(0)->dtype(), output(0)->format()};
    return megdnn_opr()->get_workspace_in_bytes(diff, data1, data2, grad1);
}

/* ==================== CorrelationBackwardData2 ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CorrelationBackwardData2);
MEGDNN_OPR_INIT3(CorrelationBackwardData2, "correlation_backward_data2", 1, true);

void CorrelationBackwardData2::scn_do_execute() {
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
            input(2)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output(1)));
}

size_t CorrelationBackwardData2::get_workspace_size_bytes(
        const TensorShapeArray& inp_shapes, const TensorShapeArray& out_shapes) const {
    TensorLayout diff{inp_shapes[0], input(0)->dtype(), input(0)->format()},
            data1{inp_shapes[1], input(1)->dtype(), input(1)->format()},
            data2{inp_shapes[2], input(2)->dtype(), input(2)->format()},
            grad2{out_shapes[0], output(0)->dtype(), output(0)->format()};
    return megdnn_opr()->get_workspace_in_bytes(diff, data1, data2, grad2);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
