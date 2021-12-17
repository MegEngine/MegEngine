/**
 * \file dnn/src/cuda/handle_create.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

// #include "src/common/handle_impl.h"

#include "src/cuda/adaptive_pooling/opr_impl.h"
#include "src/cuda/add_update/opr_impl.h"
#include "src/cuda/argmxx/opr_impl.h"
#include "src/cuda/argsort/opr_impl.h"
#include "src/cuda/batch_conv_bias/opr_impl.h"
#include "src/cuda/batch_normalization/opr_impl.h"
#include "src/cuda/batched_matrix_mul/opr_impl.h"
#include "src/cuda/check_non_finite/opr_impl.h"
#include "src/cuda/checksum/opr_impl.h"
#include "src/cuda/concat/opr_impl.h"
#include "src/cuda/cond_take/opr_impl.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/convolution/opr_impl.h"
#include "src/cuda/convolution3d/opr_impl.h"
#include "src/cuda/convpooling/opr_impl.h"
#include "src/cuda/correlation/opr_impl.h"
#include "src/cuda/cumsum/opr_impl.h"
#include "src/cuda/cvt_color/opr_impl.h"
#include "src/cuda/dct/opr_impl.h"
#include "src/cuda/deformable_conv/opr_impl.h"
#include "src/cuda/deformable_ps_roi_pooling/opr_impl.h"
#include "src/cuda/dot/opr_impl.h"
#include "src/cuda/dropout/opr_impl.h"
#include "src/cuda/elemwise/opr_impl.h"
#include "src/cuda/elemwise_multi_type/opr_impl.h"
#include "src/cuda/eye/opr_impl.h"
#include "src/cuda/fake_quant/opr_impl.h"
#include "src/cuda/fill/opr_impl.h"
#include "src/cuda/flip/opr_impl.h"
#include "src/cuda/gaussian_blur/opr_impl.h"
#include "src/cuda/group_local/opr_impl.h"
#include "src/cuda/images2neibs/opr_impl.h"
#include "src/cuda/indexing_multi_axis_vec/opr_impl.h"
#include "src/cuda/indexing_one_hot/opr_impl.h"
#include "src/cuda/layer_norm/opr_impl.h"
#include "src/cuda/linspace/opr_impl.h"
#include "src/cuda/local/opr_impl.h"
#include "src/cuda/local_share/opr_impl.h"
#include "src/cuda/lrn/opr_impl.h"
#include "src/cuda/lsq/opr_impl.h"
#include "src/cuda/mask_conv/opr_impl.h"
#include "src/cuda/matrix_inverse/opr_impl.h"
#include "src/cuda/matrix_mul/opr_impl.h"
#include "src/cuda/max_tensor_diff/opr_impl.h"
#include "src/cuda/mesh_indexing/opr_impl.h"
#include "src/cuda/padding/opr_impl.h"
#include "src/cuda/param_pack/opr_impl.h"
#include "src/cuda/pooling/opr_impl.h"
#include "src/cuda/powc/opr_impl.h"
#include "src/cuda/reduce/opr_impl.h"
#include "src/cuda/relayout/opr_impl.h"
#include "src/cuda/relayout_format/opr_impl.h"
#include "src/cuda/remap/opr_impl.h"
#include "src/cuda/repeat/opr_impl.h"
#include "src/cuda/resize/opr_impl.h"
#include "src/cuda/rng/opr_impl.h"
#include "src/cuda/roi_align/opr_impl.h"
#include "src/cuda/roi_copy/opr_impl.h"
#include "src/cuda/roi_pooling/opr_impl.h"
#include "src/cuda/rotate/opr_impl.h"
#include "src/cuda/separable_conv/opr_impl.h"
#include "src/cuda/separable_filter/opr_impl.h"
#include "src/cuda/sleep/opr_impl.h"
#include "src/cuda/sliding_window_transpose/opr_impl.h"
#include "src/cuda/softmax/opr_impl.h"
#include "src/cuda/split/opr_impl.h"
#include "src/cuda/svd/opr_impl.h"
#include "src/cuda/tensor_remap/opr_impl.h"
#include "src/cuda/tile/opr_impl.h"
#include "src/cuda/topk/opr_impl.h"
#include "src/cuda/tqt/opr_impl.h"
#include "src/cuda/transpose/opr_impl.h"
#include "src/cuda/type_cvt/opr_impl.h"
#include "src/cuda/warp_affine/opr_impl.h"
#include "src/cuda/warp_perspective/opr_impl.h"

namespace megdnn {
namespace cuda {

// After Adding CUDA LSTM, the declaration of CUDA Backend should be restored
// MEGDNN_FOREACH_OPR_CLASS(MEGDNN_SPECIALIZE_CREATE_OPERATOR)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvPoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBiasForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Images2NeibsForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Images2NeibsBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SlidingWindowTransposeForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SlidingWindowTransposeBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseMultiType);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AddUpdateForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RelayoutForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LocalForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LocalBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LocalBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LRNForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LRNBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ROIPoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ROIPoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpPerspectiveForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpPerspectiveBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpPerspectiveBackwardMat);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixInverse);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixMulForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BatchedMatrixMulForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SVDForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ReduceForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CondTake);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CumsumForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgmaxForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgminForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TransposeForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConcatForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SplitForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TileForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TileBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RepeatForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RepeatBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgsortForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgsortBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingRemapForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingRemapBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ChecksumForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingIncrMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MeshIndexing);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IncrMeshIndexing);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SetMeshIndexing);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BatchedMeshIndexing);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BatchedIncrMeshIndexing);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BatchedSetMeshIndexing);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Linspace);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Eye);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SleepForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(UniformRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GaussianRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GammaRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BetaRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoissonRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PermutationRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ShuffleRNGForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ShuffleRNGBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SeparableConvForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SeparableFilterForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupLocalForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupLocalBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupLocalBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Flip);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Rotate);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ROICopy);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CvtColor);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpAffine);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GaussianBlur);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Resize);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ResizeBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ParamPackConcat);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MaxTensorDiff);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MaskConvForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MaskPropagate);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Convolution3DForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Convolution3DBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Convolution3DBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DeformableConvForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DeformableConvBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DeformableConvBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DeformablePSROIPoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DeformablePSROIPoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RelayoutFormat);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TopK);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PowC);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LocalShareForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LocalShareBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LocalShareBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ROIAlignForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ROIAlignBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CorrelationForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CorrelationBackwardData1);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CorrelationBackwardData2);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BatchConvBiasForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Remap);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RemapBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RemapBackwardMat);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AdaptivePoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AdaptivePoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DctChannelSelectForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(FakeQuantForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(FakeQuantBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TQTForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TQTBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CheckNonFinite);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LSQForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LSQBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Fill);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PaddingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PaddingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LayerNormForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LayerNormBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DropoutForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(DropoutBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SoftmaxForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SoftmaxBackward);

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    megdnn_throw("unsupported cuda opr");
    return nullptr;
}

#define MEGDNN_INST_CREATE_OPERATOR(opr) \
    template std::unique_ptr<megdnn::opr> HandleImpl::create_operator();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
