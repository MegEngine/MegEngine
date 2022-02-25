// simd_macro/*_helper.h should be included before including this file.
//
// The following functions would be declared in this file:
//
// void local_xcorr_MEGDNN_SIMD_NAME(const LocalKParam &kparam);
// void local_conv_MEGDNN_SIMD_NAME(const LocalKParam &kparam);
//
#include "src/naive/local/opr_impl.h"

#include "src/common/macro_helper.h"

namespace megdnn {

using LocalKParam = naive::LocalForwardImpl::FloatNoncontigBatchKernParam;

void WITH_SIMD_SUFFIX(local_xcorr)(const LocalKParam& param)
        MEGDNN_SIMD_ATTRIBUTE_TARGET;

void WITH_SIMD_SUFFIX(local_conv)(const LocalKParam& param)
        MEGDNN_SIMD_ATTRIBUTE_TARGET;

}  // namespace megdnn

#include "src/common/macro_helper_epilogue.h"
