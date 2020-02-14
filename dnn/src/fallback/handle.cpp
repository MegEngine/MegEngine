/**
 * \file dnn/src/fallback/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"
#include "src/fallback/handle.h"

#include "src/fallback/convolution/opr_impl.h"
#include "src/fallback/elemwise/opr_impl.h"
#include "src/fallback/pooling/opr_impl.h"
#include "src/fallback/reduce/opr_impl.h"
#include "src/fallback/concat/opr_impl.h"
#include "src/fallback/split/opr_impl.h"
#include "src/fallback/tile/opr_impl.h"
#include "src/fallback/repeat/opr_impl.h"
#include "src/fallback/relayout/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"
#include "src/fallback/warp_perspective/opr_impl.h"
#include "src/fallback/type_cvt/opr_impl.h"
#include "src/fallback/group_local/opr_impl.h"
#include "src/fallback/flip/opr_impl.h"
#include "src/fallback/gaussian_blur/opr_impl.h"
#include "src/fallback/roi_copy/opr_impl.h"
#include "src/fallback/rotate/opr_impl.h"
#include "src/fallback/elemwise_multi_type/opr_impl.h"
#include "src/fallback/add_update/opr_impl.h"
#include "src/fallback/mask_conv/opr_impl.h"
#include "src/fallback/resize/opr_impl.h"
#include "src/fallback/batched_matrix_mul/opr_impl.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/powc/opr_impl.h"

namespace megdnn {
namespace fallback {

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    return naive::HandleImpl::create_operator<Opr>();
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(Convolution)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardData)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Elemwise)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Pooling)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Reduce)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Concat)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Split)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Tile)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Repeat)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RelayoutForward)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixMul)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpPerspective)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupLocal)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Flip)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GaussianBlur)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ROICopy)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Rotate)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseMultiType)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AddUpdate)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MaskConvForward)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Resize)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BatchedMatrixMul)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBias)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PowC)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
