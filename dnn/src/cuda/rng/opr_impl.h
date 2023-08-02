#pragma once

#include <curand.h>
#include "megdnn/oprs.h"
#include "src/cuda/handle.h"

namespace megdnn {
namespace cuda {

class CuRandHandle {
    curandGenerator_t m_gen;
    uint64_t m_seed;

    CuRandHandle(const CuRandHandle&) = delete;
    CuRandHandle& operator=(const CuRandHandle&) = delete;

public:
    CuRandHandle(cudaStream_t stream, uint64_t seed = 0);
    ~CuRandHandle();

    void seed(uint64_t seed);

    curandGenerator_t gen() const { return m_gen; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

class UniformRNGImpl : public UniformRNG {
    CuRandHandle m_curand_handle;

public:
    UniformRNGImpl(Handle* handle);
    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

class GaussianRNGImpl : public GaussianRNG {
    CuRandHandle m_curand_handle;

public:
    GaussianRNGImpl(Handle* handle);

    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout& layout) override;
};

class GammaRNGImpl : public GammaRNG {
    uint64_t m_seed, m_offset;
    cudaStream_t m_stream;

public:
    GammaRNGImpl(Handle* handle);

    void exec(
            _megdnn_tensor_in shape, _megdnn_tensor_in scale, _megdnn_tensor_out dst,
            _megdnn_workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    void seed(uint64_t seed) { m_seed = seed; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

class BetaRNGImpl : public BetaRNG {
    uint64_t m_seed, m_offset;
    cudaStream_t m_stream;

public:
    BetaRNGImpl(Handle* handle);

    void exec(
            _megdnn_tensor_in alpha, _megdnn_tensor_in beta, _megdnn_tensor_out dst,
            _megdnn_workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    void seed(uint64_t seed) { m_seed = seed; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

class PoissonRNGImpl : public PoissonRNG {
    uint64_t m_seed, m_offset;
    cudaStream_t m_stream;

public:
    PoissonRNGImpl(Handle* handle);

    void exec(
            _megdnn_tensor_in lam, _megdnn_tensor_out dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    void seed(uint64_t seed) { m_seed = seed; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

class MultinomialRNGImpl : public MultinomialRNG {
    uint64_t m_seed, m_offset;
    cudaStream_t m_stream;

public:
    MultinomialRNGImpl(Handle* handle);

    void exec(_megdnn_tensor_in probs, _megdnn_tensor_out dst, _megdnn_workspace)
            override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override;

    WorkspaceBundle get_workspace_bundle(
            void* ptr, const TensorLayout& src, const TensorLayout& dst);

    void seed(uint64_t seed) { m_seed = seed; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

class PermutationRNGImpl : public PermutationRNG {
    uint64_t m_seed, m_offset;
    cudaStream_t m_stream;

public:
    PermutationRNGImpl(Handle* handle);

    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout& layout) override;

    void seed(uint64_t seed) { m_seed = seed; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

class ShuffleRNGForwardImpl : public ShuffleRNGForward {
    uint64_t m_seed, m_offset;
    cudaStream_t m_stream;

public:
    using ShuffleRNGForward::ShuffleRNGForward;
    ShuffleRNGForwardImpl(Handle* handle);

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst,
            const TensorLayout& indices) override;

    void seed(uint64_t seed) { m_seed = seed; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

class ShuffleRNGBackwardImpl : public ShuffleRNGBackward {
    cudaStream_t m_stream;

public:
    using ShuffleRNGBackward::ShuffleRNGBackward;
    ShuffleRNGBackwardImpl(Handle* handle);
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class ExponentialRNGImpl : public ExponentialRNG {
    uint64_t m_seed, m_offset;
    cudaStream_t m_stream;

public:
    ExponentialRNGImpl(Handle* handle);

    void exec(
            _megdnn_tensor_in rate, _megdnn_tensor_out dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    void seed(uint64_t seed) { m_seed = seed; }

    void ensure_seed(uint64_t seed) {
        if (m_seed != seed) {
            this->seed(seed);
        }
    }
};

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
