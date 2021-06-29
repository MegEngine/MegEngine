/**
 * \file src/opr/include/megbrain/opr/dnn/sliding_window_transpose.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(SlidingWindowTransposeForward,
        intl::MegDNNOprWrapperFwd<megdnn::SlidingWindowTransposeForward>) // {

    public:
        SlidingWindowTransposeForward(VarNode *src,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar src,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
};
using SlidingWindowTranspose = SlidingWindowTransposeForward;

MGB_DEFINE_OPR_CLASS(SlidingWindowTransposeBackward,
        intl::MegDNNOprWrapperBwd<megdnn::SlidingWindowTransposeBackward>) // {

    public:
        SlidingWindowTransposeBackward(VarNode *diff, VarNode *src_for_shape,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar diff, SymbolVar src_for_shape,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}