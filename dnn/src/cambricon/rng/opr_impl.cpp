/**
 * \file dnn/src/cambricon/rng/opr_impl.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#include "src/cambricon/rng/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_types.h"
#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cambricon {

CnnlRandHandle::CnnlRandHandle(cnnlHandle_t handle, uint64_t seed) {
    cnnl_check(cnnlRandCreateGenerator(&m_gen, CNNL_RAND_RNG_MTGP32));
    cnnl_check(cnnlRandGetMTGP32StateSize(nullptr, &m_state_size));
    cnrt_check(cnrtMalloc(&m_state, m_state_size));
    cnrt_check(cnrtMemset(m_state, 0, m_state_size));
    this->seed(handle, seed);
}

CnnlRandHandle::~CnnlRandHandle() {
    cnnl_check(cnnlRandDestroyGenerator(m_gen));
    cnrt_check(cnrtFree(m_state));
}

void CnnlRandHandle::seed(cnnlHandle_t handle, uint64_t seed) {
    cnnl_check(cnnlRandSetPseudoRandomGeneratorSeed(m_gen, seed));
    cnnl_check(cnnlRandMakeMTGP32KernelState(
            handle, m_state, /*params=*/nullptr, /*kernel_params=*/nullptr, seed));
    m_seed = seed;
}

UniformRNGImpl::UniformRNGImpl(Handle* handle)
        : UniformRNG(handle), m_rand_handle(cnnl_handle(handle)) {}

void UniformRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    megdnn_assert(dst.layout.dtype == dtype::Float32(), "only float32 supported");
    m_rand_handle.ensure_seed(cnnl_handle(handle()), m_param.seed);
    auto cnnl_dtype = convert_to_cnnl_datatype(dst.layout.dtype.enumv());
    cnnl_check(cnnlRandGenerateUniform(
            cnnl_handle(this->handle()), m_rand_handle.gen(), cnnl_dtype,
            m_rand_handle.state(), dst.layout.total_nr_elems(), /*min=*/0.f,
            /*max=*/1.f, dst.raw_ptr()));
}

GaussianRNGImpl::GaussianRNGImpl(Handle* handle)
        : GaussianRNG(handle), m_rand_handle(cnnl_handle(handle)) {}

void GaussianRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    megdnn_assert(dst.layout.dtype == dtype::Float32(), "only float32 supported");
    m_rand_handle.ensure_seed(cnnl_handle(handle()), m_param.seed);
    auto cnnl_dtype = convert_to_cnnl_datatype(dst.layout.dtype.enumv());
    cnnl_check(cnnlRandGenerateNormal(
            cnnl_handle(handle()), m_rand_handle.gen(), cnnl_dtype,
            m_rand_handle.state(), dst.layout.total_nr_elems(), m_param.mean,
            m_param.std, dst.raw_ptr()));
}

}  // namespace cambricon
}  // namespace megdnn
// vim: syntax=cpp.doxygen
