#include "src/common/handle_impl.h"

#include "src/arm_common/handle.h"

#include "src/arm_common/adaptive_pooling/opr_impl.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/convolution/opr_impl.h"
#include "src/arm_common/cvt_color/opr_impl.h"
#include "src/arm_common/elemwise/opr_impl.h"
#include "src/arm_common/elemwise_multi_type/opr_impl.h"
#include "src/arm_common/local/opr_impl.h"
#include "src/arm_common/lstm/opr_impl.h"
#include "src/arm_common/lstm_cell/opr_impl.h"
#include "src/arm_common/pooling/opr_impl.h"
#include "src/arm_common/reduce/opr_impl.h"
#include "src/arm_common/resize/opr_impl.h"
#include "src/arm_common/rnn_cell/opr_impl.h"
#include "src/arm_common/separable_conv/opr_impl.h"
#include "src/arm_common/separable_filter/opr_impl.h"
#include "src/arm_common/type_cvt/opr_impl.h"
#include "src/arm_common/warp_affine/opr_impl.h"
#include "src/arm_common/warp_perspective/opr_impl.h"

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
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardData)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RNNCell)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LSTMCell)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LSTM)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AdaptivePooling)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
