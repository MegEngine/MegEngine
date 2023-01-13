#include "src/naive/handle.h"

#include "src/common/handle_impl.h"

#include "src/naive/adaptive_pooling/opr_impl.h"
#include "src/naive/add_update/opr_impl.h"
#include "src/naive/argmxx/opr_impl.h"
#include "src/naive/argsort/opr_impl.h"
#include "src/naive/batch_conv_bias/opr_impl.h"
#include "src/naive/batch_normalization/opr_impl.h"
#include "src/naive/batched_matrix_mul/opr_impl.h"
#include "src/naive/check_non_finite/opr_impl.h"
#include "src/naive/checksum/opr_impl.h"
#include "src/naive/concat/opr_impl.h"
#include "src/naive/cond_take/opr_impl.h"
#include "src/naive/conv_bias/opr_impl.h"
#include "src/naive/convolution/opr_impl.h"
#include "src/naive/convolution3d/opr_impl.h"
#include "src/naive/convpooling/opr_impl.h"
#include "src/naive/correlation/opr_impl.h"
#include "src/naive/cumsum/opr_impl.h"
#include "src/naive/cvt_color/opr_impl.h"
#include "src/naive/dct/opr_impl.h"
#include "src/naive/deformable_conv/opr_impl.h"
#include "src/naive/deformable_ps_roi_pooling/opr_impl.h"
#include "src/naive/diag/opr_impl.h"
#include "src/naive/dot/opr_impl.h"
#include "src/naive/dropout/opr_impl.h"
#include "src/naive/elemwise/opr_impl.h"
#include "src/naive/elemwise_multi_type/opr_impl.h"
#include "src/naive/eye/opr_impl.h"
#include "src/naive/fake_quant/opr_impl.h"
#include "src/naive/fill/opr_impl.h"
#include "src/naive/flip/opr_impl.h"
#include "src/naive/gaussian_blur/opr_impl.h"
#include "src/naive/group_local/opr_impl.h"
#include "src/naive/group_norm/opr_impl.h"
#include "src/naive/images2neibs/opr_impl.h"
#include "src/naive/indexing_multi_axis_vec/opr_impl.h"
#include "src/naive/indexing_one_hot/opr_impl.h"
#include "src/naive/lamb/opr_impl.h"
#include "src/naive/layer_norm/opr_impl.h"
#include "src/naive/linspace/opr_impl.h"
#include "src/naive/local/opr_impl.h"
#include "src/naive/local_share/opr_impl.h"
#include "src/naive/lrn/opr_impl.h"
#include "src/naive/lsq/opr_impl.h"
#include "src/naive/lstm/opr_impl.h"
#include "src/naive/lstm_cell/opr_impl.h"
#include "src/naive/mask_conv/opr_impl.h"
#include "src/naive/masked_fill/opr_impl.h"
#include "src/naive/matrix_inverse/opr_impl.h"
#include "src/naive/matrix_mul/opr_impl.h"
#include "src/naive/max_tensor_diff/opr_impl.h"
#include "src/naive/mesh_indexing/opr_impl.h"
#include "src/naive/norm/opr_impl.h"
#include "src/naive/padding/opr_impl.h"
#include "src/naive/param_pack/opr_impl.h"
#include "src/naive/pooling/opr_impl.h"
#include "src/naive/powc/opr_impl.h"
#include "src/naive/reduce/opr_impl.h"
#include "src/naive/region_restricted_convolution/opr_impl.h"
#include "src/naive/relayout/opr_impl.h"
#include "src/naive/relayout_format/opr_impl.h"
#include "src/naive/remap/opr_impl.h"
#include "src/naive/repeat/opr_impl.h"
#include "src/naive/resize/opr_impl.h"
#include "src/naive/rng/opr_impl.h"
#include "src/naive/rnn/opr_impl.h"
#include "src/naive/rnn_cell/opr_impl.h"
#include "src/naive/roi_align/opr_impl.h"
#include "src/naive/roi_copy/opr_impl.h"
#include "src/naive/roi_pooling/opr_impl.h"
#include "src/naive/rotate/opr_impl.h"
#include "src/naive/separable_conv/opr_impl.h"
#include "src/naive/separable_filter/opr_impl.h"
#include "src/naive/sleep/opr_impl.h"
#include "src/naive/sliding_window_transpose/opr_impl.h"
#include "src/naive/softmax/opr_impl.h"
#include "src/naive/split/opr_impl.h"
#include "src/naive/svd/opr_impl.h"
#include "src/naive/tensor_remap/opr_impl.h"
#include "src/naive/tile/opr_impl.h"
#include "src/naive/topk/opr_impl.h"
#include "src/naive/tqt/opr_impl.h"
#include "src/naive/transpose/opr_impl.h"
#include "src/naive/type_cvt/opr_impl.h"
#include "src/naive/warp_affine/opr_impl.h"
#include "src/naive/warp_perspective/opr_impl.h"

namespace megdnn {
namespace naive {
//! always for ci
static size_t g_image2d_pitch_alignment = 1;
static HandleImpl::HandleVendorType g_image2d_pitch_vendor =
        HandleImpl::HandleVendorType::NOT_SPEC;

DefaultConvolutionForwardAlgorithm HandleImpl::m_default_conv_fwd_algo;
DefaultConvolutionBackwardDataAlgorithm HandleImpl::m_default_conv_bwd_data_algo;
DefaultConvolutionBackwardFilterAlgorithm HandleImpl::m_default_conv_bwd_filter_algo;
DefaultConvBiasForwardAlgorithm HandleImpl::m_default_conv_bias_fwd_algo;
DefaultConvolution3DForwardAlgorithm HandleImpl::m_default_conv3d_fwd_algo;
DefaultConvolution3DBackwardDataAlgorithm HandleImpl::m_default_conv3d_bwd_data_algo;
DefaultConvolution3DBackwardFilterAlgorithm
        HandleImpl::m_default_conv3d_bwd_filter_algo;
DefaultBatchConvBiasForwardAlgorithm HandleImpl::m_default_batch_conv_bias_fwd_algo;
DefaultLocalShareForwardAlgorithm HandleImpl::m_default_local_share_fwd_algo;
DefaultLocalShareBackwardDataAlgorithm HandleImpl::m_default_local_share_bwd_data_algo;
DefaultLocalShareBackwardFilterAlgorithm
        HandleImpl::m_default_local_share_bwd_filter_algo;

DefaultMatrixMulAlgorithm HandleImpl::m_default_matmul_fwd_algo;
DefaultBatchedMatrixMulAlgorithm HandleImpl::m_default_batched_matmul_fwd_algo;

DefaultPoolingForwardAlgorithm HandleImpl::m_default_pooling_fwd_algo;
DefaultPoolingBackwardAlgorithm HandleImpl::m_default_pooling_bwd_algo;

HandleImpl::HandleImpl(megcoreComputingHandle_t computing_handle, HandleType type)
        : HandleImplHelper(computing_handle, type),
          m_dispatcher{megcoreGetCPUDispatcher(computing_handle)} {}

size_t HandleImpl::image2d_pitch_alignment() const {
    return g_image2d_pitch_alignment;
}

HandleImpl::HandleVendorType HandleImpl::vendor_type() const {
    return g_image2d_pitch_vendor;
}

size_t HandleImpl::exchange_image2d_pitch_alignment(size_t alignment) {
    auto ret = g_image2d_pitch_alignment;
    g_image2d_pitch_alignment = alignment;
    return ret;
}

HandleImpl::HandleVendorType HandleImpl::exchange_image2d_vendor(
        HandleImpl::HandleVendorType vendor) {
    auto ret = g_image2d_pitch_vendor;
    g_image2d_pitch_vendor = vendor;
    return ret;
}

MEGDNN_FOREACH_OPR_CLASS(MEGDNN_SPECIALIZE_CREATE_OPERATOR)

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
