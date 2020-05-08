/**
 * \file dnn/src/arm_common/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"

#include "src/arm_common/handle.h"

#include "src/arm_common/convolution/opr_impl.h"
#include "src/arm_common/pooling/opr_impl.h"
#include "src/arm_common/local/opr_impl.h"
#include "src/arm_common/separable_conv/opr_impl.h"
#include "src/arm_common/separable_filter/opr_impl.h"
#include "src/arm_common/elemwise/opr_impl.h"
#include "src/arm_common/elemwise_multi_type/opr_impl.h"
#include "src/arm_common/cvt_color/opr_impl.h"
#include "src/arm_common/warp_affine/opr_impl.h"
#include "src/arm_common/resize/opr_impl.h"
#include "src/arm_common/warp_perspective/opr_impl.h"
#include "src/arm_common/type_cvt/opr_impl.h"
#include "src/arm_common/reduce/opr_impl.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/winograd_filter_preprocess/opr_impl.h"

namespace megdnn {
namespace arm_common {

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    return fallback::HandleImpl::create_operator<Opr>();
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(Pooling)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Local)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SeparableConv)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SeparableFilter)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Elemwise)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseMultiType)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CvtColor)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpAffine)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Resize)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpPerspective)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Reduce)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBias)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WinogradFilterPreprocess)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardData)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

} // namespace arm_common
} // namespace megdnn

// vim: syntax=cpp.doxygen
