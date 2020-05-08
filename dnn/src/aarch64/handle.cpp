/**
 * \file dnn/src/aarch64/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"

#include "src/aarch64/handle.h"
#include "src/aarch64/matrix_mul/opr_impl.h"
#include "src/aarch64/rotate/opr_impl.h"
#include "src/aarch64/relayout/opr_impl.h"
#include "src/aarch64/conv_bias/opr_impl.h"
#include "src/aarch64/warp_perspective/opr_impl.h"

namespace megdnn {
namespace aarch64 {

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    return arm_common::HandleImpl::create_operator<Opr>();
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixMul)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Rotate)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RelayoutForward)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBias)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpPerspective)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

} // namespace aarch64
} // namespace megdnn

// vim: syntax=cpp.doxygen
