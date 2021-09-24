/**
 * \file src/gopt/impl/opr_format_modifier.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megbrain/graph.h"
#include "megbrain/opr/dnn/convolution.h"

namespace mgb {
namespace gopt {
namespace intl {

#define FOREACH_FORMAT_AWARE_OPR(cb)                                                   \
    cb(Convolution) cb(ConvBiasForward) cb(ConvolutionBackwardData) cb(PoolingForward) \
            cb(WarpPerspective) cb(Resize)
bool has_available_algo(const VarNodeArray& i, const cg::OperatorNodeBase* opr);

VarNode* modify_opr_format(
        opr::ConvBias::Param::Format opr_format, const VarNodeArray& i,
        const cg::OperatorNodeBase* opr);

}  // namespace intl
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
