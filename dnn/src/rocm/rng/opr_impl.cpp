/**
 * \file dnn/src/rocm/rng/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/common/utils.h"
#include "src/rocm/handle.h"
#include "src/rocm/utils.h"
#include "./opr_impl.h"

using namespace megdnn;
using namespace rocm;

namespace {
const char* status2str(rocrand_status status) {
    switch (status) {
#define C(v) \
    case v:  \
        return #v
        C(ROCRAND_STATUS_SUCCESS);
        C(ROCRAND_STATUS_VERSION_MISMATCH);
        C(ROCRAND_STATUS_NOT_CREATED);
        C(ROCRAND_STATUS_ALLOCATION_FAILED);
        C(ROCRAND_STATUS_TYPE_ERROR);
        C(ROCRAND_STATUS_OUT_OF_RANGE);
        C(ROCRAND_STATUS_LENGTH_NOT_MULTIPLE);
        C(ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED);
        C(ROCRAND_STATUS_LAUNCH_FAILURE);
        C(ROCRAND_STATUS_INTERNAL_ERROR);
#undef C
    }
    return "unknown";
}
#define ROCRAND_CHECK(expr)                                                \
    do {                                                                   \
        rocrand_status status = (expr);                                    \
        MEGDNN_MARK_USED_VAR(&status2str);                                 \
        if (status != ROCRAND_STATUS_SUCCESS) {                            \
            megdnn_throw(                                                  \
                    ssprintf("rocrand call failed: status=%d(%s) call=%s", \
                             status, status2str(status), #expr));          \
        }                                                                  \
    } while (0)

}  // namespace

RocRANDHandle::RocRANDHandle(hipStream_t stream, uint64_t seed) {
    ROCRAND_CHECK(rocrand_create_generator(&m_gen, ROCRAND_RNG_PSEUDO_XORWOW));
    ROCRAND_CHECK(rocrand_set_stream(m_gen, stream));
    this->seed(seed);
}

RocRANDHandle::~RocRANDHandle() {
    if (rocrand_destroy_generator(m_gen) != ROCRAND_STATUS_SUCCESS) {
        megdnn_trap();
    }
}

void RocRANDHandle::seed(uint64_t seed) {
    ROCRAND_CHECK(rocrand_set_seed(m_gen, seed));
    m_seed = seed;
}

UniformRNGImpl::UniformRNGImpl(Handle *handle):
    UniformRNG(handle),
    m_rocrand_handle(hip_stream(handle))
{
}

void UniformRNGImpl::exec(
        _megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    megdnn_assert(dst.layout.dtype == dtype::Float32(),
            "only float32 supported");
    m_rocrand_handle.ensure_seed(m_param.seed);
    ROCRAND_CHECK(rocrand_generate_uniform(m_rocrand_handle.gen(),
                dst.ptr<dt_float32>(), dst.layout.total_nr_elems()));
}

GaussianRNGImpl::GaussianRNGImpl(Handle *handle):
    GaussianRNG(handle),
    m_rocrand_handle(hip_stream(handle))
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
    m_rocrand_handle.ensure_seed(m_param.seed);
    auto gen = m_rocrand_handle.gen();
    if (size % 2) {
        auto wk = workspace.ptr<dt_float32>();
        ROCRAND_CHECK(rocrand_generate_normal(gen, wk, 2, m_param.mean,
                    m_param.std));
        hip_check(hipMemcpyAsync(
                    ptr + size - 1, wk, sizeof(dt_float32), hipMemcpyDeviceToDevice,
                    hip_stream(handle())));
        -- size;
    }

    if (size) {
        ROCRAND_CHECK(rocrand_generate_normal(
                    gen, ptr, size, m_param.mean, m_param.std));
    }
}

size_t GaussianRNGImpl::get_workspace_in_bytes(const TensorLayout &layout) {
    if (layout.total_nr_elems() % 2)
        return 2 * layout.dtype.size();
    return 0;
}

// vim: syntax=cpp.doxygen

