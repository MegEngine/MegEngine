#pragma once

#include "megdnn/handle.h"
#include "megdnn/oprs.h"

#include "src/common/utils.h"

#include <mutex>

#include "midout.h"

MIDOUT_DECL(dnn_src_common_handle_impl)

namespace megdnn {

class HandleImplHelper : public Handle {
public:
    using Handle::Handle;

    //! global matmul opr
    virtual MatrixMul* matmul_opr() { megdnn_throw("Unimplement matmul opr.\n"); }

    //! global matmul opr with first operand transposed
    virtual MatrixMul* matmul_aT_opr() { megdnn_throw("Unimplement matmul_aT opr.\n"); }

    //! global matmul opr with second operand transposed
    virtual MatrixMul* matmul_bT_opr() { megdnn_throw("Unimplement matmul_bT opr.\n"); }

    //! global matmul opr with both operand transposed
    virtual MatrixMul* matmul_aT_bT_opr() {
        megdnn_throw("Unimplement matmul_aT_bT opr.\n");
    }

    //! global relayout opr
    virtual Relayout* relayout_opr() { megdnn_throw("Unimplement Relayout opr.\n"); }

    virtual Checksum* checksum_opr() { megdnn_throw("Unimplement Checksum opr.\n"); }

    virtual MaxTensorDiff* max_tensor_diff_opr() {
        megdnn_throw("Unimplement MaxTensorDiff opr.\n");
    }

protected:
    static constexpr size_t NR_HELPER_OPRS = 7;

    template <class Opr, size_t idx, class Self>
    static Opr* get_helper_opr(Self self, const typename Opr::Param& param = {}) {
        MIDOUT_BEGIN(dnn_src_common_handle_impl, Opr, idx) {
            static_assert(idx < NR_HELPER_OPRS, "invalid idx");
            if (!self->m_helper_oprs[idx]) {
                MEGDNN_LOCK_GUARD(self->m_helper_oprs_mtx);
                if (!self->m_helper_oprs[idx]) {
                    self->m_helper_oprs[idx] = self->template create_operator<Opr>();
                    auto ret = static_cast<Opr*>(self->m_helper_oprs[idx].get());
                    ret->param() = param;
                    megdnn_assert(ret->is_thread_safe());
                    return ret;
                }
            }
            return static_cast<Opr*>(self->m_helper_oprs[idx].get());
        }
        MIDOUT_END();
    }

private:
    std::array<std::unique_ptr<OperatorBase>, NR_HELPER_OPRS> m_helper_oprs;
    DNN_MUTEX m_helper_oprs_mtx;
};

}  // namespace megdnn
   /*!
    * \brief iterate though each operator class name; useful for explicit
    *      instantialization of create_operator<> templates
    */
   // clang-format off
#define MEGDNN_FOREACH_OPR_CLASS(cb) \
    cb(ConvolutionForward) \
    cb(ConvolutionBackwardData) \
    cb(ConvolutionBackwardFilter) \
    cb(ConvPoolingForward) \
    cb(ConvBiasForward) \
    cb(Images2NeibsForward) \
    cb(Images2NeibsBackward) \
    cb(SlidingWindowTransposeForward) \
    cb(SlidingWindowTransposeBackward) \
    cb(ElemwiseForward) \
    cb(ElemwiseMultiType) \
    cb(AddUpdateForward) \
    cb(RelayoutForward) \
    cb(PoolingForward) \
    cb(PoolingBackward) \
    cb(LocalForward) \
    cb(LocalBackwardData) \
    cb(LocalBackwardFilter) \
    cb(LRNForward) \
    cb(LRNBackward) \
    cb(ROIPoolingForward) \
    cb(ROIPoolingBackward) \
    cb(WarpPerspectiveForward) \
    cb(WarpPerspectiveBackwardData) \
    cb(WarpPerspectiveBackwardMat) \
    cb(DotForward) \
    cb(MatrixInverse) \
    cb(MatrixMulForward) \
    cb(BatchedMatrixMulForward) \
    cb(SVDForward) \
    cb(ReduceForward) \
    cb(CondTake) \
    cb(CumsumForward) \
    cb(ArgmaxForward) \
    cb(ArgminForward) \
    cb(TransposeForward) \
    cb(ConcatForward) \
    cb(SplitForward) \
    cb(TileForward) \
    cb(TileBackward) \
    cb(RepeatForward) \
    cb(RepeatBackward) \
    cb(ArgsortForward) \
    cb(ArgsortBackward) \
    cb(TypeCvt) \
    cb(IndexingRemapForward) \
    cb(IndexingRemapBackward) \
    cb(ChecksumForward) \
    cb(IndexingOneHotForward) \
    cb(IndexingSetOneHotForward) \
    cb(IndexingMultiAxisVec) \
    cb(IndexingSetMultiAxisVec) \
    cb(IndexingIncrMultiAxisVec) \
    cb(MeshIndexing) \
    cb(IncrMeshIndexing) \
    cb(SetMeshIndexing) \
    cb(BatchedMeshIndexing) \
    cb(BatchedIncrMeshIndexing) \
    cb(BatchedSetMeshIndexing) \
    cb(Linspace) \
    cb(Eye) \
    cb(Diag) \
    cb(SleepForward) \
    cb(UniformRNG) \
    cb(GaussianRNG) \
    cb(GammaRNG) \
    cb(BetaRNG) \
    cb(PoissonRNG) \
    cb(PermutationRNG) \
    cb(ShuffleRNGForward) \
    cb(ShuffleRNGBackward) \
    cb(SeparableConvForward) \
    cb(SeparableFilterForward) \
    cb(BNForward) \
    cb(BNBackward) \
    cb(GroupLocalForward) \
    cb(GroupLocalBackwardData) \
    cb(GroupLocalBackwardFilter) \
    cb(Flip) \
    cb(Rotate) \
    cb(ROICopy) \
    cb(CvtColor) \
    cb(WarpAffine) \
    cb(GaussianBlur) \
    cb(Resize) \
    cb(ResizeBackward) \
    cb(ParamPackConcat) \
    cb(MaxTensorDiff) \
    cb(MaskConvForward) \
    cb(MaskPropagate) \
    cb(Convolution3DForward) \
    cb(Convolution3DBackwardData) \
    cb(Convolution3DBackwardFilter) \
    cb(DeformableConvForward) \
    cb(DeformableConvBackwardFilter) \
    cb(DeformableConvBackwardData) \
    cb(DeformablePSROIPoolingForward) \
    cb(DeformablePSROIPoolingBackward) \
    cb(RelayoutFormat) \
    cb(TopK) \
    cb(PowC) \
    cb(LocalShareForward) \
    cb(LocalShareBackwardData) \
    cb(LocalShareBackwardFilter) \
    cb(ROIAlignForward) \
    cb(ROIAlignBackward) \
    cb(CorrelationForward) \
    cb(CorrelationBackwardData1) \
    cb(CorrelationBackwardData2) \
    cb(BatchConvBiasForward) \
    cb(Remap) \
    cb(RemapBackwardData) \
    cb(RemapBackwardMat) \
    cb(AdaptivePoolingForward) \
    cb(AdaptivePoolingBackward) \
    cb(DctChannelSelectForward) \
    cb(FakeQuantForward) \
    cb(FakeQuantBackward) \
    cb(TQTForward) \
    cb(TQTBackward) \
    cb(CheckNonFinite) \
    cb(LSQForward) \
    cb(LSQBackward) \
    cb(Fill) \
    cb(PaddingForward) \
    cb(PaddingBackward) \
    cb(LayerNormForward) \
    cb(LayerNormBackward) \
    cb(GeneralNormForward) \
    cb(GeneralNormBackward) \
    cb(DropoutForward) \
    cb(DropoutBackward) \
    cb(RNNCell) \
    cb(LSTMCell) \
    cb(RNN) \
    cb(RNNBackward) \
    cb(LSTM) \
    cb(LAMBUpdate) \
    cb(LSTMBackward) \
    cb(SoftmaxForward) \
    cb(SoftmaxBackward) \
    cb(NormForward) \
    cb(RegionRestrictedConvolutionForward) \
    cb(RegionRestrictedConvolutionBackwardData) \
    cb(RegionRestrictedConvolutionBackwardFilter) \
    cb(GroupNormForward) \
    cb(GroupNormBackward) \
    cb(MaskedFill) \
    cb(MultiHeadAttnForward)\
    cb(MultiHeadAttnBackward)
// clang-format on

/*!
 * \brief specialize HandleImpl::create_operator for a single opr type;
 *      implemented by <opr>Impl class
 */
#define MEGDNN_SPECIALIZE_CREATE_OPERATOR(opr)                   \
    template <>                                                  \
    std::unique_ptr<megdnn::opr> HandleImpl::create_operator() { \
        return megdnn::make_unique<opr##Impl>(this);             \
    }

/*!
 * \brief for explicit instantiation for HandleImpl::create_operator methods
 */
#define MEGDNN_INST_CREATE_OPERATOR(opr) \
    template std::unique_ptr<megdnn::opr> HandleImpl::create_operator();

// vim: syntax=cpp.doxygen
