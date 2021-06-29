/**
 * \file src/opr/impl/dnn/sliding_window_transpose.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/sliding_window_transpose.h"
#include "megbrain/graph/grad_impl.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SlidingWindowTransposeForward);
MEGDNN_OPR_INIT1(SlidingWindowTransposeForward, "sliding_window_transpose")

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(SlidingWindowTransposeForward) {
    mgb_assert(wrt_idx == 0 && out_grad.size() == 2 && !out_grad[1]);
    return SlidingWindowTransposeBackward::make(
            out_grad[0], opr.input(0), opr.param()).node();
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SlidingWindowTransposeBackward);
MEGDNN_OPR_INIT2(SlidingWindowTransposeBackward, "sliding_window_transpose_grad", 1, false);


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

