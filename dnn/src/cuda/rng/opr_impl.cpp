/**
 * \file dnn/src/cuda/rng/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"
#include "./opr_impl.h"

using namespace megdnn;
using namespace cuda;

namespace {
    const char *status2str(curandStatus_t status) {
        switch (status) {
#define C(v) case v: return #v
            C(CURAND_STATUS_SUCCESS);
            C(CURAND_STATUS_VERSION_MISMATCH);
            C(CURAND_STATUS_NOT_INITIALIZED);
            C(CURAND_STATUS_ALLOCATION_FAILED);
            C(CURAND_STATUS_TYPE_ERROR);
            C(CURAND_STATUS_OUT_OF_RANGE);
            C(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
            C(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
            C(CURAND_STATUS_LAUNCH_FAILURE);
            C(CURAND_STATUS_PREEXISTING_FAILURE);
            C(CURAND_STATUS_INITIALIZATION_FAILED);
            C(CURAND_STATUS_ARCH_MISMATCH);
            C(CURAND_STATUS_INTERNAL_ERROR);
#undef C
        }
        return "unknown";
    }
#define CURAND_CHECK(expr) \
    do { \
            curandStatus_t status = (expr); \
            MEGDNN_MARK_USED_VAR(&status2str); \
            if (status != CURAND_STATUS_SUCCESS) { \
                megdnn_throw(ssprintf( \
                            "curand call failed: status=%d(%s) call=%s", \
                            status, status2str(status), # expr)); \
        } \
    } while (0)

} // anonymouse namespace

CuRandHandle::CuRandHandle(cudaStream_t stream, uint64_t seed) {
    CURAND_CHECK(curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(m_gen, stream));
    this->seed(seed);
}

CuRandHandle::~CuRandHandle() {
    if (curandDestroyGenerator(m_gen) != CURAND_STATUS_SUCCESS) {
        megdnn_trap();
    }
}

void CuRandHandle::seed(uint64_t seed) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(m_gen, seed));
    m_seed = seed;
}

UniformRNGImpl::UniformRNGImpl(Handle *handle):
    UniformRNG(handle),
    m_curand_handle(cuda_stream(handle))
{
}

void UniformRNGImpl::exec(
        _megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    megdnn_assert(dst.layout.dtype == dtype::Float32(),
            "only float32 supported");
    m_curand_handle.ensure_seed(m_param.seed);
    CURAND_CHECK(curandGenerateUniform(m_curand_handle.gen(),
                dst.ptr<dt_float32>(), dst.layout.total_nr_elems()));
}

GaussianRNGImpl::GaussianRNGImpl(Handle *handle):
    GaussianRNG(handle),
    m_curand_handle(cuda_stream(handle))
{
}

void GaussianRNGImpl::exec(
        _megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    megdnn_assert(dst.layout.dtype == dtype::Float32(),
            "only float32 supported");
    auto ptr = dst.ptr<dt_float32>();
    auto size = dst.layout.total_nr_elems();
    megdnn_assert(size);
    m_curand_handle.ensure_seed(m_param.seed);
    auto gen = m_curand_handle.gen();
    if (size % 2) {
        auto wk = workspace.ptr<dt_float32>();
        CURAND_CHECK(curandGenerateNormal(gen, wk, 2, m_param.mean,
                    m_param.std));
        cuda_check(cudaMemcpyAsync(
                    ptr + size - 1, wk, sizeof(dt_float32), cudaMemcpyDeviceToDevice,
                    cuda_stream(handle())));
        -- size;
    }

    if (size) {
        CURAND_CHECK(curandGenerateNormal(
                    gen, ptr, size, m_param.mean, m_param.std));
    }
}

size_t GaussianRNGImpl::get_workspace_in_bytes(const TensorLayout &layout) {
    if (layout.total_nr_elems() % 2)
        return 2 * layout.dtype.size();
    return 0;
}

// vim: syntax=cpp.doxygen

