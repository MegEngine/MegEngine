/**
 * \file src/opr/impl/dnn/fake_quant.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/dnn/fake_quant.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(FakeQuantForward);
MEGDNN_OPR_INIT3(FakeQuantForward, "fakequant_fwd");

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(FakeQuantForward) {
    if (wrt_idx == 0) {
        // wrt src
        SymbolVar grad =
                FakeQuantBackward::make(out_grad[0], opr.input(0), opr.input(1),
                                        opr.input(2), opr.param());
        return grad.node();
    } else {
        return nullptr;
    }
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(FakeQuantBackward);
MEGDNN_OPR_INIT4(FakeQuantBackward, "fakequant_bwd", 1, true);
