/**
 * \file src/opr/impl/dnn/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"

namespace mgb {
namespace opr {
namespace intl {

    //! grad for conv like oprs: Fwd should take (src, filter) as its param
    template<class BwdData, class BwdFilter, class Fwd>
    VarNode* conv_grad(
            Fwd &opr, size_t wrt_idx, const VarNodeArray &out_grad) {
        mgb_assert(wrt_idx == 0 || wrt_idx == 1);
        mgb_assert(out_grad.size() == 2);
        if (wrt_idx == 0) {
            // data
            SymbolVar grad = BwdData::make(
                    opr.input(1), out_grad[0], opr.input(0), opr.param());
            return grad.node();
        } else {
            // filter
            SymbolVar grad = BwdFilter::make(
                    opr.input(0), out_grad[0], opr.input(1), opr.param());
            return grad.node();
        }
    }

} // namespace intl
} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

