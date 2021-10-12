/**
 * \file dnn/src/naive/rng/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstdint>
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

//! see http://xoroshiro.di.unimi.it/splitmix64.c
class Splitmix64 {
    uint64_t m_s;

public:
    explicit Splitmix64(uint64_t seed = 0) : m_s{seed} {}

    uint64_t operator()();
};

/*!
 * \brief the xoroshiro+ PRNG described at http://xoroshiro.di.unimi.it/
 */
class Xoroshiro128plus {
    uint64_t m_s[2], m_init_seed = 0;
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    explicit Xoroshiro128plus(uint64_t seed = 0) { this->seed(seed); }

    //! reset state if seed changed
    Xoroshiro128plus& ensure_seed(uint64_t seed) {
        if (seed != m_init_seed) {
            this->seed(seed);
        }
        return *this;
    }

    //! set seed
    void seed(uint64_t seed);

    uint64_t operator()();
};

class UniformRNGImpl : public UniformRNG {
    Xoroshiro128plus m_rng;

public:
    using UniformRNG::UniformRNG;
    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

class GaussianRNGImpl : public GaussianRNG {
    Xoroshiro128plus m_rng;

public:
    using GaussianRNG::GaussianRNG;
    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

class GammaRNGImpl : public GammaRNG {
    Xoroshiro128plus m_rng;

public:
    using GammaRNG::GammaRNG;

    void exec(
            _megdnn_tensor_in shape, _megdnn_tensor_in scale, _megdnn_tensor_out dst,
            _megdnn_workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class PoissonRNGImpl : public PoissonRNG {
    Xoroshiro128plus m_rng;

public:
    using PoissonRNG::PoissonRNG;

    void exec(_megdnn_tensor_in lam, _megdnn_tensor_inout dst, _megdnn_workspace)
            override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class BetaRNGImpl : public BetaRNG {
    Xoroshiro128plus m_rng;

public:
    using BetaRNG::BetaRNG;

    void exec(
            _megdnn_tensor_in alpha, _megdnn_tensor_in beta, _megdnn_tensor_out dst,
            _megdnn_workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class PermutationRNGImpl : public PermutationRNG {
    Xoroshiro128plus m_rng;

public:
    using PermutationRNG::PermutationRNG;

    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

class ShuffleRNGForwardImpl : public ShuffleRNGForward {
    Xoroshiro128plus m_rng;

public:
    using ShuffleRNGForward::ShuffleRNGForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class ShuffleRNGBackwardImpl : public ShuffleRNGBackward {
    Xoroshiro128plus m_rng;

public:
    using ShuffleRNGBackward::ShuffleRNGBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
