/**
 * \file dnn/test/common/opr_trait.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

#include <cstddef>

namespace megdnn {
namespace test {

template <typename Opr>
struct OprTrait {};

#define DEF(Name, Arity, HasWorkspace, CanDeduceLayout)        \
    template <>                                                \
    struct OprTrait<Name> {                                    \
        static const size_t arity = Arity;                     \
        static const bool has_workspace = HasWorkspace;        \
        static const bool can_deduce_layout = CanDeduceLayout; \
    }

DEF(ConvolutionForward, 3, true, true);
DEF(Convolution3DForward, 3, true, true);
DEF(ConvolutionBackwardData, 3, true, false);
DEF(ConvolutionBackwardFilter, 3, true, false);
DEF(Convolution3DBackwardData, 3, true, false);
DEF(Convolution3DBackwardFilter, 3, true, false);
DEF(ConvPoolingForward, 4, true, true);
DEF(ConvBiasForward, 5, true, true);
DEF(SeparableConvForward, 4, true, true);
DEF(SeparableFilterForward, 4, true, true);
DEF(Images2NeibsForward, 2, true, true);
DEF(Images2NeibsBackward, 2, true, false);
DEF(PoolingForward, 2, true, true);
DEF(PoolingBackward, 4, true, false);
DEF(AdaptivePoolingForward, 2, true, false);
DEF(AdaptivePoolingBackward, 4, true, false);
DEF(LocalForward, 3, true, true);
DEF(LocalBackwardData, 3, true, false);
DEF(LocalBackwardFilter, 3, true, false);
DEF(GroupLocalForward, 3, true, true);
DEF(GroupLocalBackwardData, 3, true, false);
DEF(GroupLocalBackwardFilter, 3, true, false);
DEF(LRNForward, 2, true, true);
DEF(LRNBackward, 4, true, false);
DEF(BNForward, 8, true, true);
DEF(BNBackward, 8, true, false);
DEF(ROIPoolingForward, 4, true, false);
DEF(ROIPoolingBackward, 5, true, false);
DEF(WarpPerspectiveForward, 3, true, false);
DEF(WarpPerspectiveBackwardData, 3, true, false);
DEF(WarpPerspectiveBackwardMat, 4, true, false);
DEF(AddUpdateForward, 2, false, false);
DEF(DotForward, 3, true, true);
DEF(MatrixMulForward, 3, true, true);
DEF(BatchedMatrixMulForward, 3, true, true);
DEF(MatrixInverse, 2, true, true);
DEF(SVDForward, 4, true, true);
DEF(ReduceForward, 2, true, true);
DEF(CumsumForward, 2, true, true);
DEF(ArgmaxForward, 2, true, true);
DEF(ArgminForward, 2, true, true);
DEF(TransposeForward, 2, true, true);
DEF(RelayoutForward, 2, false, false);
DEF(TileForward, 2, true, true);
DEF(TileBackward, 2, true, false);
DEF(RepeatForward, 2, true, true);
DEF(RepeatBackward, 2, true, false);
DEF(ArgsortForward, 3, true, true);
DEF(ArgsortBackward, 3, true, false);
DEF(TypeCvtForward, 2, false, false);
DEF(IndexingRemapForward, 3, true, true);
DEF(IndexingRemapBackward, 3, true, false);
DEF(Linspace, 1, true, false);
DEF(Eye, 1, true, false);
DEF(Flip, 2, true, true);
DEF(ROICopy, 2, true, true);
DEF(Rotate, 2, true, true);
DEF(CvtColor, 2, true, true);
DEF(WarpAffine, 3, true, false);
DEF(GaussianBlur, 2, true, true);
DEF(Resize, 2, true, false);
DEF(ResizeBackward, 2, true, false);
DEF(IndexingOneHot, 3, true, true);
DEF(IndexingSetOneHot, 3, true, false);
DEF(MaskConvolution, 4, true, true);
DEF(MaskPropagate, 2, true, true);
DEF(RelayoutFormat, 2, true, true);
DEF(MaxTensorDiff, 2, true, false);
DEF(WinogradFilterPreprocess, 2, true, true);
DEF(LocalShareForward, 3, true, true);
DEF(LocalShareBackwardData, 3, true, false);
DEF(LocalShareBackwardFilter, 3, true, false);
DEF(ROIAlignForward, 4, true, true);
DEF(ROIAlignBackward, 4, true, false);
DEF(DeformableConvForward, 5, true, true);
DEF(DeformableConvBackwardFilter, 5, true, false);
DEF(DeformableConvBackwardData, 8, true, false);
DEF(DeformablePSROIPoolingForward, 5, true, true);
DEF(DeformablePSROIPoolingBackward, 7, true, false);
DEF(BatchConvBiasForward, 5, true, true);
DEF(Remap, 3, true, true);
DEF(RemapBackwardData, 3, true, false);
DEF(RemapBackwardMat, 4, true, false);
DEF(DctChannelSelectForward, 4, true, true);
DEF(FakeQuantForward, 4, true, true);
DEF(FakeQuantBackward, 5, true, false);
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
