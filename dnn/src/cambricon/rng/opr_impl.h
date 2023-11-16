/**
 * \file dnn/src/cambricon/rng/opr_impl.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megdnn/oprs.h"
#include "src/cambricon/handle.h"

namespace megdnn {
namespace cambricon {

class CnnlRandHandle {
    cnnlRandGenerator_t m_gen;
    uint64_t m_seed;
    size_t m_state_size;
    void* m_state;

    CnnlRandHandle(const CnnlRandHandle&) = delete;
    CnnlRandHandle& operator=(const CnnlRandHandle&) = delete;

public:
    CnnlRandHandle(cnnlHandle_t handle, uint64_t seed = 0);
    ~CnnlRandHandle();

    void seed(cnnlHandle_t handle, uint64_t seed);
    void* state() const { return m_state; }
    cnnlRandGenerator_t gen() const { return m_gen; }
    void ensure_seed(cnnlHandle_t handle, uint64_t seed) {
        if (m_seed != seed) {
            this->seed(handle, seed);
        }
    }
};

class UniformRNGImpl : public UniformRNG {
    CnnlRandHandle m_rand_handle;

public:
    UniformRNGImpl(Handle* handle);
    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

class GaussianRNGImpl : public GaussianRNG {
    CnnlRandHandle m_rand_handle;

public:
    GaussianRNGImpl(Handle* handle);

    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

}  // namespace cambricon
}  // namespace megdnn
// vim: syntax=cpp.doxygen
