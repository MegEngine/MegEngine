/**
 * \file dnn/src/x86/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"
#include "src/common/version_symbol.h"

#include "src/x86/handle.h"

#include "src/x86/add_update/opr_impl.h"
#include "src/x86/conv_bias/opr_impl.h"
#include "src/x86/cvt_color/opr_impl.h"
#include "src/x86/elemwise/opr_impl.h"
#include "src/x86/elemwise_multi_type/opr_impl.h"
#include "src/x86/gaussian_blur/opr_impl.h"
#include "src/x86/local/opr_impl.h"
#include "src/x86/lrn/opr_impl.h"
#include "src/x86/matrix_mul/opr_impl.h"
#include "src/x86/pooling/opr_impl.h"
#include "src/x86/resize/opr_impl.h"
#include "src/x86/separable_conv/opr_impl.h"
#include "src/x86/separable_filter/opr_impl.h"
#include "src/x86/type_cvt/opr_impl.h"
#include "src/x86/utils.h"
#include "src/x86/warp_affine/opr_impl.h"
#include "src/x86/warp_perspective/opr_impl.h"

#if MEGDNN_X86_WITH_MKL

#include <mkl.h>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define MKL_VERSION_STR                   \
    STR(__INTEL_MKL__)                    \
    "." STR(__INTEL_MKL_MINOR__) "." STR( \
            __INTEL_MKL_UPDATE__) " (build date " STR(__INTEL_MKL_BUILD_DATE) ")"
#pragma message "compile with Intel MKL " MKL_VERSION_STR "."
#endif

namespace megdnn {
namespace x86 {

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    return fallback::HandleImpl::create_operator<Opr>();
}

HandleImpl::HandleImpl(megcoreComputingHandle_t computing_handle,
                       HandleType type)
        : fallback::HandleImpl::HandleImpl(computing_handle, type) {
    disable_denorm();
#if MEGDNN_X86_WITH_MKL
    vmlSetMode(VML_LA | VML_FTZDAZ_ON | VML_ERRMODE_ERRNO);
#endif

#if MEGDNN_X86_WITH_MKL_DNN
    m_mkldnn_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    m_mkldnn_stream = dnnl::stream(m_mkldnn_engine);
#endif
}

size_t HandleImpl::alignment_requirement() const {
    // AVX-512 requires 64byte alignment; we use this max value here
    return 64;
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(SeparableConv)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SeparableFilter)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Pooling)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Local)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(LRN)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixMul)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Elemwise)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseMultiType)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CvtColor)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpAffine)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GaussianBlur)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Resize)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WarpPerspective)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AddUpdate)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBias)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace x86
}  // namespace megdnn

MEGDNN_VERSION_SYMBOL3(MKL, __INTEL_MKL__, __INTEL_MKL_MINOR__,
                       __INTEL_MKL_UPDATE__);

// vim: syntax=cpp.doxygen
