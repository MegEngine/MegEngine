#pragma once
#include "megdnn/internal/opr_header_prologue.h"

namespace megdnn {

//! base class for random number generators
class RNGBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(RNGBase, OperatorBase);

public:
    virtual void exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& dst) = 0;

protected:
    virtual void check_exec(const TensorLayout& dst, size_t workspace_in_bytes) = 0;
};

//! sample from poisson distribution
class PoissonRNG : public OperatorBase {
    DEF_OPR_IMPL(PoissonRNG, OperatorBase, 1, 1);
    DEF_OPR_PARAM(PoissonRNG);

public:
    virtual void exec(
            _megdnn_tensor_in lam, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& lam, const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& lam, const TensorLayout& dst,
            size_t workspace_in_bytes);
};

//! sample from beta distribution
class BetaRNG : public OperatorBase {
    DEF_OPR_IMPL(BetaRNG, OperatorBase, 2, 1);
    DEF_OPR_PARAM(BetaRNG);

public:
    virtual void exec(
            _megdnn_tensor_in alpha, _megdnn_tensor_in beta, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& alpha, const TensorLayout& beta,
            const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& alpha, const TensorLayout& beta,
            const TensorLayout& dst, size_t workspace_in_bytes);
};

//! sample from gamma distribution
class GammaRNG : public OperatorBase {
    DEF_OPR_IMPL(GammaRNG, OperatorBase, 2, 1);
    DEF_OPR_PARAM(GammaRNG);

public:
    virtual void exec(
            _megdnn_tensor_in shape, _megdnn_tensor_in scale, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& shape, const TensorLayout& scale,
            const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& shape, const TensorLayout& scale,
            const TensorLayout& dst, size_t workspace_in_bytes);
};

//! sample from uniform distribution on the interval (0, 1]
class UniformRNG : public RNGBase {
    DEF_OPR_IMPL(UniformRNG, RNGBase, 0, 1);
    DEF_OPR_PARAM(UniformRNG);

protected:
    void check_exec(const TensorLayout& dst, size_t workspace_in_bytes);
};

//! sample from gaussian distribution
class GaussianRNG : public RNGBase {
    DEF_OPR_IMPL(GaussianRNG, RNGBase, 0, 1);
    DEF_OPR_PARAM(GaussianRNG);

protected:
    void check_exec(const TensorLayout& dst, size_t workspace_in_bytes);
};

class PermutationRNG : public RNGBase {
    DEF_OPR_IMPL(PermutationRNG, RNGBase, 0, 1);
    DEF_OPR_PARAM(PermutationRNG);

protected:
    void check_exec(const TensorLayout& dst, size_t workspace_in_bytes);
};

class ShuffleRNGForward : public OperatorBase {
    DEF_OPR_IMPL(ShuffleRNGForward, OperatorBase, 1, 2);
    DEF_OPR_PARAM(ShuffleRNG);

public:
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& src, TensorLayout& dst, TensorLayout& indices);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst,
            const TensorLayout& indices) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& dst,
            const TensorLayout& indices, size_t workspace_in_bytes);
};
using ShuffleRNG = ShuffleRNGForward;

class ShuffleRNGBackward : public OperatorBase {
    DEF_OPR_IMPL(ShuffleRNGBackward, OperatorBase, 2, 1);
    DEF_OPR_PARAM(ShuffleRNG);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& indices,
            const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& indices,
            const TensorLayout& grad, size_t workspace_in_bytes);
};

//! sample from exponential distribution
class ExponentialRNG : public OperatorBase {
    DEF_OPR_IMPL(ExponentialRNG, OperatorBase, 1, 1);
    DEF_OPR_PARAM(ExponentialRNG);

public:
    virtual void exec(
            _megdnn_tensor_in rate, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& rate, const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& rate, const TensorLayout& dst,
            size_t workspace_in_bytes);
};

/*!
 * \brief sleep for specific time on the computing device; useful for testing
 *      async problems
 */
class SleepForward : public OperatorBase {
    DEF_OPR_IMPL(SleepForward, OperatorBase, 0, 0);
    DEF_OPR_PARAM(Sleep);

public:
    virtual void exec() = 0;
};
using Sleep = SleepForward;

/*!
 * \brief calculating checksum of a tensor
 *
 * data must be a one-dimensional contiguous tensor with dtype byte
 */
class ChecksumForward : public OperatorBase {
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL(ChecksumForward, OperatorBase, 0, 1);

public:
    using Result = opr_result::Checksum;

    virtual size_t get_workspace_in_bytes(const TensorLayout& data) = 0;

    virtual Result exec(_megdnn_tensor_in data, _megdnn_workspace workspace) = 0;

protected:
    void check_exec(const TensorLayout& layout, size_t workspace_in_bytes);
};
using Checksum = ChecksumForward;

/*!
 * \brief calculating max absolute difference of the two input tensors
 *
 * src1 and src2 must be a one-dimensional contiguous tensor.
 */
class MaxTensorDiff : public OperatorBase {
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL(MaxTensorDiff, OperatorBase, 0, 2);

public:
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& layout1, const TensorLayout& layout2) = 0;

    virtual float exec(
            _megdnn_tensor_in src1, _megdnn_tensor_in src2,
            _megdnn_workspace workspace) = 0;

protected:
    void check_exec(
            const TensorLayout& layout1, const TensorLayout& layout2,
            size_t workspace_in_bytes);
};

bool check_bias_share_in_channel(
        const TensorLayout& bias, const param::ConvBias::Format format);

}  // namespace megdnn

#include "megdnn/internal/opr_header_epilogue.h"

// vim: syntax=cpp.doxygen
