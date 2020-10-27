/**
 * \file src/opr/impl/dnn/lrn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/lrn.h"
#include "megbrain/graph/grad_impl.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LRNForward);
MEGDNN_OPR_INIT1(LRNForward, "lrn")

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LRNForward) {
    mgb_assert(wrt_idx == 0);
    SymbolVar grad = LRNBackward::make(
            opr.input(0), opr.output(0), out_grad[0], opr.param());
    return grad.node();
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LRNBackward);
MEGDNN_OPR_INIT3(LRNBackward, "lrn_bwd", 0, true);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

