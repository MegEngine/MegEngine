#pragma once
#include "megdnn/oprs.h"

#include <cstddef>

namespace megdnn {

template <typename Opr>
struct OprTrait {};

#define DEF(Name, Arity, HasWorkspace, CanDeduceLayout)        \
    template <>                                                \
    struct OprTrait<Name> {                                    \
        static const size_t arity = Arity;                     \
        static const bool has_workspace = HasWorkspace;        \
        static const bool can_deduce_layout = CanDeduceLayout; \
    }

DEF(Norm, 2, true, true);
DEF(Padding, 2, false, true);
DEF(PaddingBackward, 2, false, false);
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
DEF(SlidingWindowTransposeForward, 2, true, true);
DEF(SlidingWindowTransposeBackward, 2, true, false);
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
DEF(BNForward, 9, true, true);
DEF(BNBackward, 9, true, false);
DEF(ROIPoolingForward, 4, true, false);
DEF(ROIPoolingBackward, 5, true, false);
DEF(CorrelationForward, 3, true, true);
DEF(CorrelationBackwardData1, 4, true, true);
DEF(CorrelationBackwardData2, 4, true, true);
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
DEF(Diag, 2, true, true);
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
DEF(TQTForward, 3, true, true);
DEF(TQTBackward, 5, true, false);
DEF(PowC, 2, false, true);
DEF(UniformRNG, 1, true, true);
DEF(GaussianRNG, 1, true, true);
DEF(GammaRNG, 3, true, true);
DEF(BetaRNG, 3, true, true);
DEF(PoissonRNG, 2, true, true);
DEF(PermutationRNG, 1, true, true);
DEF(ShuffleRNGForward, 3, true, true);
DEF(ShuffleRNGBackward, 3, true, false);
DEF(ChecksumForward, 1, true, false);
DEF(CheckNonFinite, 2, true, true);
DEF(LSQForward, 5, true, true);
DEF(LSQBackward, 7, true, false);
DEF(Fill, 1, true, false);
DEF(LayerNormForward, 6, true, true);
DEF(LayerNormBackward, 8, true, true);
DEF(LAMBUpdate, 7, true, true);
DEF(DropoutForward, 3, true, true);
DEF(DropoutBackward, 3, true, true);
DEF(RNNCellForward, 7, true, true);
DEF(RNNForward, 6, true, true);
DEF(RNNBackward, 10, true, true);
DEF(LSTMCellForward, 10, true, true);
DEF(LSTMForward, 8, true, true);
DEF(LSTMBackward, 13, true, true);
DEF(SoftmaxForward, 2, true, true);
DEF(SoftmaxBackward, 3, true, false);
}  // namespace megdnn

// vim: syntax=cpp.doxygen
