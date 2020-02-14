/**
 * \file src/opr/include/megbrain/opr/dnn/images2neibs.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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

MGB_DEFINE_OPR_CLASS(Images2NeibsForward,
        intl::MegDNNOprWrapperFwd<megdnn::Images2NeibsForward>) // {

    public:
        Images2NeibsForward(VarNode *src,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar src,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
};
using Images2Neibs = Images2NeibsForward;

MGB_DEFINE_OPR_CLASS(Images2NeibsBackward,
        intl::MegDNNOprWrapperBwd<megdnn::Images2NeibsBackward>) // {

    public:
        Images2NeibsBackward(VarNode *diff, VarNode *src_for_shape,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar diff, SymbolVar src_for_shape,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
