/**
 * \file src/opr/impl/dnn/tqt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/dnn/tqt.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TQTForward);
MEGDNN_OPR_INIT2(TQTForward, "tqt_fwd");

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(TQTForward) {
    SymbolVarArray grad = TQTBackward::make(out_grad[0], opr.input(0),
                                            opr.input(1), opr.param());

    if (wrt_idx == 0) {
        return grad[0].node();
    } else if (wrt_idx == 1) {
        return reduce_sum(grad[1], GetVarShape::make(opr.input(wrt_idx)))
                .node();
    } else {
        return nullptr;
    }
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TQTBackward);

TQTBackward::TQTBackward(VarNode* y_grad, VarNode* x, VarNode* scale,
                         const Param& param, const OperatorNodeConfig& config)
        : Super({x->owner_graph(), config, "tqt_bwd", {y_grad, x, scale}}, 1,
                true) {
    init_megdnn_opr(*this, param);
    add_input({y_grad, x, scale});
}

SymbolVarArray TQTBackward::make(SymbolVar y_grad, SymbolVar x, SymbolVar scale,
                                 const Param& param,
                                 const OperatorNodeConfig& config) {
    auto&& out = x.node()->owner_graph()
                         ->insert_opr(std::make_unique<TQTBackward>(
                                 y_grad.node(), x.node(), scale.node(), param,
                                 config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = out[i];
    }
    return ret;
}

void TQTBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    mgr.register_shape_infer(output(0),
                             ShapeInferDesc::make_identity(input(1)));
    mgr.register_shape_infer(output(1),
                             ShapeInferDesc::make_identity(input(1)));
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::TQTBackward>::val);
}

void TQTBackward::init_output_dtype() {
    output(0)->dtype(input(1)->dtype());
    output(1)->dtype(input(2)->dtype());
}
