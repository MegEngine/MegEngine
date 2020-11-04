/**
 * \file dnn/src/rocm/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/common/handle_impl.h"
#include "src/common/version_symbol.h"

#include "src/rocm/handle.h"
#include "src/rocm/miopen_with_check.h"
#include "src/rocm/utils.h"

#include "src/rocm/checksum/opr_impl.h"
#include "src/rocm/convolution/opr_impl.h"
#include "src/rocm/elemwise/opr_impl.h"
#include "src/rocm/eye/opr_impl.h"
#include "src/rocm/pooling/opr_impl.h"
#include "src/rocm/reduce/opr_impl.h"
#include "src/rocm/type_cvt/opr_impl.h"
#include "src/rocm/add_update/opr_impl.h"
#include "src/rocm/matrix_mul/opr_impl.h"
#include "src/rocm/batched_matrix_mul/opr_impl.h"
#include "src/rocm/indexing_one_hot/opr_impl.h"
#include "src/rocm/rng/opr_impl.h"
#include "src/rocm/relayout/opr_impl.h"
#include "src/rocm/powc/opr_impl.h"
#include "src/rocm/indexing_multi_axis_vec/opr_impl.h"
#include "src/rocm/linspace/opr_impl.h"
#include "src/rocm/argmxx/opr_impl.h"
#include "src/rocm/sleep/opr_impl.h"
#include "src/rocm/batch_normalization/opr_impl.h"

#include <miopen/version.h>
#include <hip/hip_version.h>

#include <cstring>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define MIOPEN_VERSION_STR    \
    STR(MIOPEN_VERSION_MAJOR) \
    "." STR(MIOPEN_VERSION_MINOR) "." STR(MIOPEN_VERSION_PATCH)

#pragma message "compile with MIOpen " MIOPEN_VERSION_STR " "

#undef STR
#undef STR_HELPER

namespace megdnn {
std::unique_ptr<Handle> Handle::make_rocm_handle(megcoreComputingHandle_t computing_handle) {
    return std::make_unique<rocm::HandleImpl>(computing_handle);
}
template <typename Opr>
std::unique_ptr<Opr> Handle::create_rocm_operator() {
    return static_cast<rocm::HandleImpl*>(this)->create_operator<Opr>();
}
#define INST(opr) \
    template std::unique_ptr<opr> Handle::create_rocm_operator();
MEGDNN_FOREACH_OPR_CLASS(INST)
#undef INST
}

namespace megdnn {
namespace rocm {

HandleImpl::HandleImpl(megcoreComputingHandle_t comp_handle)
        : HandleImplHelper(comp_handle, HandleType::ROCM) {
    // Get megcore device handle
    megcoreDeviceHandle_t dev_handle;
    megcoreGetDeviceHandle(comp_handle, &dev_handle);
    int dev_id;
    megcoreGetDeviceID(dev_handle, &dev_id);
    if (dev_id < 0) {
        hip_check(hipGetDevice(&dev_id));
    }
    m_device_id = dev_id;
    hip_check(hipGetDeviceProperties(&m_device_prop, dev_id));
    // Get stream from MegCore computing handle.
    //! no version check
    megcore::getROCMContext(comp_handle, &m_megcore_context);
    rocblas_check(rocblas_create_handle(&m_rocblas_handle));
    //! must call miopenCreateWithStream() to create miopen handle, then the
    //! rocblas_handle of miopen will set to be the same stream , otherwise
    //! miopen create rocblas_handle with default stream
    miopen_check(miopenCreateWithStream(&m_miopen_handle, stream()));

    // Set stream for miopen and rocblas handles.
    rocblas_check(rocblas_set_stream(m_rocblas_handle, stream()));

    // Note that all rocblas scalars (alpha, beta) and scalar results such as
    // dot output resides at device side.
    rocblas_check(rocblas_set_pointer_mode(m_rocblas_handle,
                                           rocblas_pointer_mode_device));

    // init const scalars
    hip_check(hipMalloc(&m_const_scalars, sizeof(ConstScalars)));
    ConstScalars const_scalars_val;
    const_scalars_val.init();
    hip_check(hipMemcpyAsync(m_const_scalars, &const_scalars_val,
                             sizeof(ConstScalars), hipMemcpyHostToDevice,
                             stream()));
    hip_check(hipStreamSynchronize(stream()));
}

HandleImpl::~HandleImpl() noexcept {
    miopen_check(miopenDestroy(m_miopen_handle));
    rocblas_check(rocblas_destroy_handle(m_rocblas_handle));
    hip_check(hipFree(m_const_scalars));
}

void HandleImpl::ConstScalars::init() {
#if !MEGDNN_DISABLE_FLOAT16
    f16[0].megdnn_x = 0;
    f16[1].megdnn_x = 1;
#endif
    f32[0] = 0;
    f32[1] = 1;
    i32[0] = 0;
    i32[1] = 1;
}

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    megdnn_throw("unsupported rocm opr");
    return nullptr;
}

size_t HandleImpl::alignment_requirement() const {
    auto&& prop = m_device_prop;
    MEGDNN_MARK_USED_VAR(prop);
    //! for now, texture functions are not supported.
    return 1u;
}

bool HandleImpl::check_cross_dev_copy_constraint(const TensorLayout& src) {
    // is contiguous or can be hold by
    // relayout::param::try_copy_2d/try_copy_last_contig
    return src.is_contiguous() || src.stride[src.ndim - 1] == 1;
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Eye);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ChecksumForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ReduceForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AddUpdateForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixMulForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BatchedMatrixMulForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(UniformRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GaussianRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RelayoutForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PowC);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingIncrMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Linspace);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgmaxForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgminForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SleepForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNBackward);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace rocm
}  // namespace megdnn


MEGDNN_VERSION_SYMBOL3(HIP, HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH);
MEGDNN_VERSION_SYMBOL3(MIOPEN, MIOPEN_VERSION_MAJOR, MIOPEN_VERSION_MINOR,
                       MIOPEN_VERSION_PATCH);
// vim: syntax=cpp.doxygen
