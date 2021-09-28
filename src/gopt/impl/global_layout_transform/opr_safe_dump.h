/**
 * \file src/gopt/impl/global_layout_transform/opr_safe_dump.h
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

namespace mgb {
namespace gopt {
namespace intl {
#define FOREACH_SUPPORTED_OPR(cb)                                          \
    cb(Convolution) cb(ConvBiasForward) cb(ConvolutionBackwardData)        \
            cb(PoolingForward) cb(WarpPerspective) cb(Resize) cb(Elemwise) \
                    cb(ElemwiseMultiType) cb(Concat) cb(PowC) cb(TypeCvt)

std::string opr_safe_dump(const cg::OperatorNodeBase* opr);

}  // namespace intl
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
