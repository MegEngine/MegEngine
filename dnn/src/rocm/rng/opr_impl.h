/**
 * \file dnn/src/rocm/rng/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/oprs.h"
#include "src/rocm/handle.h"
#include <rocrand.h>

namespace megdnn {
namespace rocm {

class RocRANDHandle {
    rocrand_generator m_gen;
    uint64_t m_seed;

    RocRANDHandle(const RocRANDHandle&) = delete;
    RocRANDHandle& operator = (const RocRANDHandle&) = delete;

    public:
        RocRANDHandle(hipStream_t stream, uint64_t seed = 0);
        ~RocRANDHandle();

        void seed(uint64_t seed);

        rocrand_generator gen() const {
            return m_gen;
        }

        void ensure_seed(uint64_t seed) {
            if (m_seed != seed) {
                this->seed(seed);
            }
        }
};

class UniformRNGImpl: public UniformRNG {
    RocRANDHandle m_rocrand_handle;

    public:
        UniformRNGImpl(Handle *handle);
        void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

        size_t get_workspace_in_bytes(const TensorLayout&) override {
            return 0;
        }
};

class GaussianRNGImpl: public GaussianRNG {
    RocRANDHandle m_rocrand_handle;

    public:
        GaussianRNGImpl(Handle *handle);
        void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

        size_t get_workspace_in_bytes(const TensorLayout &layout) override;
};


} // namespace rocm
} // namespace megdnn
// vim: syntax=cpp.doxygen

